import scann
import torch
import logging
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import src.utils.autograd_hacks as autograd_hacks

from collections import OrderedDict
from src.utils.simclr import info_nce_loss
from src.hbird_eval import hbird_evaluation
from src.utils.grads import get_layer, get_layer_grads_dim
from src.utils.misc import seed_everything, model_feature_dim, configure_logging
from src.utils.compression import generate_projection_matrix, suggested_scaling_factor


def main(args):
    logging.info(f"the script arguments are {args}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.hub.load("facebookresearch/dino:main", args.model).to(device)

    E = model_feature_dim(model=model, device=device, image_size=args.input_size)

    model = nn.Sequential(OrderedDict([
        ("backbone", model),
        ("fc", nn.Linear(in_features=E, out_features=args.latent_dim))
    ]))

    # Load the gradients projection matrix
    if args.projection_matrix:
        logging.info(f"loading the projection matrix from {args.projection_matrix}")

        projection_data = torch.load(args.projection_matrix)

        scaling = projection_data["scaling"]
        projection = projection_data["projection"].to(device)
    else:
        logging.info("no projection matrix was specified. Creating it from scratch")

        grads_dim = get_layer_grads_dim(model=model, path=args.gradients_layer)
        grads_dim = torch.prod(torch.tensor(grads_dim))

        projection = generate_projection_matrix(
            dims=(E, grads_dim),
            device=device
        )

        scaling = suggested_scaling_factor(projection_dim=E)

    if args.use_fp16:
        projection = projection.to(torch.bfloat16)

    scaler = torch.cuda.amp.GradScaler(enabled=args.use_fp16)
    gradient_paths = autograd_hacks.preprocess_layer_paths(
        paths=[args.gradients_layer]
    )

    # Add hooks to compute per-patch gradients
    autograd_hacks.add_hooks(model, layer_paths=gradient_paths)

    logging.info(f"loading the patches memory bank from {args.memory_bank}")

    memory = torch.load(args.memory_bank, map_location=torch.device("cpu"))

    logging.info(f"loading the patches scann index from {args.scann_index}")

    voc_index = scann.scann_ops_pybind.load_searcher(args.scann_index)

    logging.info("done loading memory and index")

    def combined_token_features(model, imgs):
        autograd_hacks.clear_backprops(model)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=args.use_fp16):
            # Compute the tokens for the last transformer block
            # including the CLS token
            tokens = model.backbone.get_intermediate_layers(imgs)[0]

            B, T, E = tokens.shape

            # Project the tokens to the latent dimension using the projection head
            latent_tokens = model.fc(tokens.reshape(B * T, E))
            latent_tokens = latent_tokens.reshape(B, T, -1)

            # Tokens are now [B, T, L], where T is the number of tokens
            # and L is the output dimension of the projection head
            B, T, _ = latent_tokens.shape

            # Retrieve N neighbors for every token
            neighbor_indices, _ = voc_index.search_batched(
                tokens.reshape(B * T, E).detach().cpu().numpy()
            )

            neighbors = memory[torch.tensor(neighbor_indices.astype(np.int64))]

            # Reshape the neighbors so they have shape [B, T, N, E], where N
            # is the number of neighbors pulled for each image token
            neighbors = neighbors.reshape(B, T, E, -1).permute(0, 1, 3, 2)

            # Only select N neighbors for the negatives batch
            neighbors = neighbors[:, :, :args.num_negative_neighbors, :]

            _, _, N, _ = neighbors.shape

            # Project the neighbors using the same projection head,
            # but exclude them from the gradients computation
            with torch.no_grad():
                neighbors = neighbors.to(device)
                neighbors = model.fc(neighbors.reshape(B * T * N, -1))
                neighbors = neighbors.reshape(B, T, N, -1)

            # Calculate a loss for each batch item by contrasting
            # the image tokens against the fetched neighbors
            losses = torch.zeros(B).to(device)

            for i in range(B):
                batch_tokens = latent_tokens[i].to(device)

                # Only the neighbors actually used for the
                # loss computation are moved to the GPU
                if args.negative_batch_percent >= 1.0:
                    comparison_batch = neighbors[i].reshape(T * N, -1).to(device)
                else:
                    # Sample a subset of the negative batch for the loss computation
                    comparison_batch = neighbors[i].reshape(T * N, -1)
                    comparison_batch_mask = torch.rand(
                        comparison_batch.shape[0]
                    ) <= args.negative_batch_percent

                    comparison_batch = comparison_batch[comparison_batch_mask, :].to(device)

                loss = info_nce_loss(
                    comparison_batch_features=comparison_batch,
                    positive_features=batch_tokens
                )

                losses[i] = loss

            loss = losses.mean()

        if args.use_fp16:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Calculate the per-token gradients
        autograd_hacks.compute_grad1(model, layer_paths=gradient_paths)

        layer = get_layer(model, args.gradients_layer)
        dtype = torch.bfloat16 if args.use_fp16 else torch.float32

        # The batch gradients have shape [B, T, W, H + 1]
        batch_gradients = torch.cat([
            layer.weight.grad1.to(dtype),
            layer.bias.grad1.unsqueeze(dim=-1).to(dtype)
        ], dim=-1)

        # Subtract one as we're getting rid of the CLS token
        T = T - 1

        # Remove the CLS token because it does not correspond
        # to a real image patch
        batch_gradients = batch_gradients[:, 1:, :, :]
        batch_gradients = batch_gradients.reshape(B * T, -1)

        tokens = tokens[:, 1:, :]

        # Project the gradients from [H, W + 1] to E and reshape
        # them back to per token gradients
        token_gradients = scaling * (projection @ batch_gradients.T).permute(1, 0)
        token_gradients = token_gradients.reshape(B, T, E)

        # Normalize the embeddings and gradients independently
        tokens = F.normalize(tokens, dim=-1, p=2)
        token_gradients = F.normalize(token_gradients, dim=-1, p=2)

        # Concatenate tokens and gradients and normalize the new features
        augmented_features = torch.cat([tokens, token_gradients], dim=-1)
        augmented_features = F.normalize(augmented_features, dim=-1, p=2)

        return augmented_features, None

    hbird_miou = hbird_evaluation(
        model.to(device),
        # size of the embedding feature vectors of patches. Here we double
        # the embeddings size as we are also using the gradients information
        d_model=E * 2,
        patch_size=args.patch_size,
        batch_size=args.batch_size,
        input_size=args.input_size,
        augmentation_epoch=args.augmentation_epochs,
        device=device,
        return_knn_details=False,
        # The number of neighbors to fetch per image patch
        num_neighbour=args.num_neighbors,
        # Other parameters to be used for the k-NN operator
        nn_params={
            "num_leaves": args.num_leaves,
            "num_leaves_to_search": args.num_leaves_to_search,
            "anisotropic_quantization_threshold": args.anisotropic_quantization_threshold,
            "num_reordering_candidates": args.num_reordering_candidates,
            "dimensions_per_block": args.dimensions_per_block
        },
        # The function that maps an image to patch features
        ftr_extr_fn=combined_token_features,
        dataset_name="voc",
        data_dir=args.data_dir,
        memory_size=args.memory_size,
        num_train_samples=args.num_train_samples,
        temperature=args.temperature
    )

    logging.info(f"Hummingbird Evaluation (mIoU): {hbird_miou}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HummingBird Evaluation")

    # Standard arguments
    parser.add_argument("--seed", default=42, type=int, help="The seed for the random number generators")
    parser.add_argument("--use-fp16", action=argparse.BooleanOptionalAction, default=False, help="Whether to run the model in fp16")

    # Model arguments
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--input-size", type=int, default=224, help="Size of the images fed to the model")
    parser.add_argument("--patch-size", type=int, default=16, help="Size of the model patches")
    parser.add_argument("--num-negative-neighbors", type=int, default=5, help="The number of neighbors used for the contrastive loss")
    parser.add_argument("--negative-batch-percent", type=float, default=1, help="The percentage of the negative batch samples to be used for the loss computation")
    parser.add_argument("--memory-size", type=int, default=None, help="The size of the memory bank")
    parser.add_argument("--model", type=str, required=True, help="DINO model name")
    parser.add_argument("--augmentation-epochs", type=int, default=1, help="The number of augmentation epochs")
    parser.add_argument("--latent-dim", type=int, required=True, help="The latent dim of the projection head")
    parser.add_argument("--temperature", type=float, required=False, default=0.02, help="The cross-attention temperature")

    # NN index parameters
    parser.add_argument("--num-neighbors", type=int, default=30, help="The number of neighbors retrieved from the NN index")
    parser.add_argument("--num-leaves", type=int, default=512, help="The number of leaves for the NN index")
    parser.add_argument("--num-leaves-to-search", type=int, default=32, help="The number of leaves to search for the NN index")
    parser.add_argument("--num-reordering-candidates", type=int, default=120, help="The number of candidates to rerank for the NN index")
    parser.add_argument("--anisotropic-quantization-threshold", type=float, default=0.2, help="The anisotropic quantization threshold for the NN index")
    parser.add_argument("--dimensions-per-block", type=int, default=4, help="The dimensions per block for the NN index")

    # Gradients and output arguments
    parser.add_argument("--projection-matrix", type=str, required=False, default=None, help="Path to the projection matrix for gradients")
    parser.add_argument("--gradients-layer", type=str, required=True, help="The layer from which gradients will be extracted from")

    # Data arguments
    parser.add_argument("--data-dir", type=str, default="VOCSegmentation", help="Path to the VOC dataset")
    parser.add_argument("--memory-bank", type=str, required=True, help="Path to the patches memory bank")
    parser.add_argument("--scann-index", type=str, required=True, help="Path to the patches memory bank scann index")
    parser.add_argument("--num-train-samples", type=int, required=False, default=None, help="How many dataset samples should be used? None means all samples will be used")

    args = parser.parse_args()

    configure_logging()
    seed_everything(args.seed)
    main(args)
