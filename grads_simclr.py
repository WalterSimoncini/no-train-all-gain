import h5py
import wandb
import torch
import pickle
import logging
import argparse
import torch.nn as nn
import src.utils.autograd_hacks as autograd_hacks

from tqdm import tqdm
from torch.utils.data import DataLoader

from src.models import get_model
from src.utils.grads import get_layer
from src.datasets import load_dataset
from src.utils.seed import seed_everything
from src.utils.logging import configure_logging
from src.utils.args import parse_gradient_targets
from src.enums import DatasetSplit, ModelType, DatasetType
from src.utils.models import freeze_model, freeze_batchnorm_modules, model_feature_dim
from src.utils.simclr import (
    sample_batch,
    info_nce_loss,
    get_simclr_transform,
    SimCLRAugmentationType,
    precompute_comparison_batch
)


class SimCLRHead(nn.Module):
    def __init__(self, backbone: nn.Module, embeddings_dim: int, output_dim: int):
        super().__init__()

        self.backbone = backbone
        self.projection = nn.Linear(embeddings_dim, output_dim)

    def forward(self, x):
        x = self.backbone(x)

        return self.projection(x)


def main(args):
    wandb.init(
        project="knnfun",
        entity="walter-simoncini"
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, _ = get_model(type_=args.model, cache_dir=args.cache_dir, pretrained=True)
    model = model.to(device)

    embedding_size = model_feature_dim(
        model=model,
        device=device,
        image_size=args.input_size
    )

    model = SimCLRHead(
        backbone=model,
        embeddings_dim=embedding_size,
        output_dim=args.latent_dim
    ).to(device)

    transform = get_simclr_transform(
        n_views=args.n_views,
        augmentation_type=args.augmentation_type,
        stride_scale=args.stride_scale,
        view_size=args.input_size
    )

    logging.info(f"freezing batch norm modules...")

    freeze_batchnorm_modules(model, device=device)

    # When patchify is used we use the same number of views for both positive
    # and negative samples so that the "receptive fields" are equal
    if args.augmentation_type == SimCLRAugmentationType.PATCHIFY:
        num_negative_views = args.n_views
    else:
        num_negative_views = 2

    negatives_transform = get_simclr_transform(
        n_views=num_negative_views,
        augmentation_type=args.augmentation_type,
        stride_scale=args.stride_scale,
        view_size=args.input_size
    )

    comparison_dataset = load_dataset(
        type_=args.comparison_batch_dataset,
        split=args.comparison_batch_dataset_split,
        cache_dir=args.cache_dir,
        transform=negatives_transform
    )

    logging.info(f"the data transform is: {transform}")
    logging.info(f"fixed data augmentation: {args.fixed_augmentation}")
    logging.info(f"the comparison batch size is: {args.comparison_batch_size}")
    logging.info(f"the model is {type(model.backbone)} (the model argument was: {args.model})")
    logging.info(f"the projection head is {model.projection}")

    model = model.to(device)

    # Load the datasets and the comparison batch, including the
    # optional hand-picked hard negatives. The final comparison
    # batch will always have the same shape (the free space
    # will be filled with randomly sampled hard negatives)
    if args.hard_negatives_path is not None:
        hard_negatives = pickle.load(open(args.hard_negatives_path, "rb"))
        num_hard_negatives = len(hard_negatives)

        transformed_hard_negatives = []

        for negative in hard_negatives:
            # Generate multiple views for each hard negative
            negative_views = [x.unsqueeze(0) for x in negatives_transform(negative)]
            transformed_hard_negatives.extend(negative_views)

        hard_negatives = torch.cat(transformed_hard_negatives, dim=0)
        hard_negatives = precompute_comparison_batch(
            batch=hard_negatives,
            model=model,
            device=device
        )

        logging.info(f"loaded {num_hard_negatives} hard negatives from {args.hard_negatives_path}")

        assert num_hard_negatives <= args.comparison_batch_size, "The hard-negatives dataset can be at most as big as the comparison batch!"
    else:
        num_hard_negatives = 0

    # Fill in the remaining slots in the comparison batch,
    # precompute the features and merge them with the hard
    # negatives if any.
    comparison_batch = sample_batch(
        dataset=comparison_dataset,
        batch_size=args.comparison_batch_size - num_hard_negatives
    )

    comparison_batch = precompute_comparison_batch(
        batch=comparison_batch,
        model=model,
        device=device
    )

    logging.info(f"the comparison batch has shape: {comparison_batch.shape}")

    if num_hard_negatives > 0:
        comparison_batch = torch.cat([
            hard_negatives,
            comparison_batch
        ], dim=0)

    logging.info(f"the final comparison batch (including hard negatives) has shape {comparison_batch.shape}")

    # FIXME: There is probably a way to calculate the loss(es)
    # for multiple samples at once, but we're using a small
    # model so this does not matter much (yet)
    data_loader = DataLoader(
        load_dataset(
            type_=args.dataset,
            split=args.dataset_split,
            cache_dir=args.cache_dir,
            transform=transform
        ),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True
    )

    # Load the model
    if args.checkpoint:
        logging.info(f"loading checkpoint from {args.checkpoint}")

        model.load_state_dict(
            torch.load(args.checkpoint, map_location=device)["state_dict"]
        )
    else:
        logging.info("no checkpoint was specified: loading a raw model")

    if args.gradients_layers is not None:
        gradients_layers = parse_gradient_targets(targets=args.gradients_layers)

        # Freeze all modules except the ones we want to extract gradients from
        freeze_model(model, exclusions=list(gradients_layers.keys()))
    else:
        raise ValueError(f"You must specify at least one mapping in --gradients-layers")

    # Load the random projection matrix
    projection_data = torch.load(args.projection_matrix)

    scaling = projection_data["scaling"]
    projection = projection_data["projection"].to(device)
    feature_dim = projection.shape[0]

    if args.use_fp16:
        projection = projection.to(torch.bfloat16)

    grads_file = h5py.File(args.output_path, "w")

    for layer_path, dataset_name in gradients_layers.items():
        grads_file.create_dataset(dataset_name, (len(data_loader.dataset), feature_dim), dtype="float32")

    # Iterate over the dataset, and for each sample calculate
    # the InfoNCE loss, backpropagate and extract the gradients
    # from the target layers.
    logging.info(f"extracting gradients...")

    # Set the same seed for every epoch and step so that different
    # samples receive the same data augmentation.
    if args.fixed_augmentation:
        seed_everything(args.seed)

    scaler = torch.cuda.amp.GradScaler(enabled=args.use_fp16)
    gradient_paths = autograd_hacks.preprocess_layer_paths(
        paths=list(gradients_layers.keys())
    )

    # Add hooks to calculate the per-sample gradients
    autograd_hacks.add_hooks(model, layer_paths=gradient_paths)

    for batch_index, (test_views, _) in tqdm(enumerate(data_loader)):
        _, P, _, _, _ = test_views.shape

        actual_batch_size = test_views.shape[0]
        batch_offset = args.batch_size * batch_index

        loss = step(
            model=model,
            test_views=test_views,
            comparison_batch=comparison_batch,
            device=device,
            use_fp16=args.use_fp16,
            scaler=scaler,
            negatives_mixing=args.negatives_mixing,
            negatives_mixing_k=args.negatives_mixing_k,
            temperature=args.temperature
        )

        # Compute the per-sample gradients
        autograd_hacks.compute_grad1(model, layer_paths=gradient_paths)

        # Log the current step to wandb so we can also observe
        # some system metrics (e.g. GPU memory and resource
        # utilization)
        wandb.log({
            "step": batch_index,
            "loss": loss
        })

        for layer_path, dataset_name in gradients_layers.items():
            layer = get_layer(model=model, path=layer_path)

            # This code works regardless of the batch size (even for batch size = 1)
            if type(layer) == nn.Conv2d:
                batch_gradients = layer.weight.grad1.reshape(actual_batch_size, P, -1).sum(dim=1)
            else:
                H, W = layer.weight.shape[:2]

                weight_gradient = layer.weight.grad1.reshape(actual_batch_size, P, H, W).sum(dim=1)

                if hasattr(layer, "bias") and layer.bias is not None:
                    # Extract the gradient for the bias vector as well
                    bias_gradient = layer.bias.grad1.sum(dim=1).reshape(actual_batch_size, P, -1).sum(dim=1).unsqueeze(dim=-1)
                    batch_gradients = torch.cat([weight_gradient, bias_gradient], dim=-1)
                else:
                    batch_gradients = weight_gradient

            if args.use_fp16:
                batch_gradients = batch_gradients.to(torch.bfloat16)

            batch_gradients = batch_gradients.view(actual_batch_size, -1)
            batch_gradients = scaling * (projection @ batch_gradients.T).permute(1, 0)

            grads_file[dataset_name][batch_offset:batch_offset + actual_batch_size] = batch_gradients.to(torch.float32).cpu().numpy()

        if args.fixed_augmentation:
            seed_everything(args.seed)

    grads_file.close()

    logging.info(f"saved the gradients to {args.output_path}")


def step(
    model: nn.Module,
    test_views: torch.Tensor,
    comparison_batch: torch.Tensor,
    device: torch.device,
    use_fp16: bool = False,
    scaler: torch.cuda.amp.GradScaler = None,
    negatives_mixing: bool = False,
    negatives_mixing_k: int = None,
    temperature: float = 0.07
) -> torch.Tensor:
    model.zero_grad()

    # Clear tensors used for the per-sample gradient computation
    autograd_hacks.clear_backprops(model)

    # Calculate the batch size for this iteration, as the last batch
    # may have a smaller size than regular batches
    actual_batch_size = test_views.shape[0]

    _, P, C, H, W = test_views.shape

    features = test_views.reshape(-1, C, H, W).to(device)

    with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_fp16):
        features = model(features)
        features = features.reshape(actual_batch_size, P, -1)

        losses = torch.zeros(features.shape[0]).to(device)

        for i, sample_features in enumerate(features):
            losses[i] = info_nce_loss(
                comparison_batch_features=comparison_batch,
                positive_features=sample_features,
                device=device,
                mixing=negatives_mixing,
                mixing_k=negatives_mixing_k,
                temperature=temperature
            )

        loss = losses.mean()

        assert not torch.isnan(loss), "found a NaN loss. Stopping the gradient extraction"

    if use_fp16:
        scaler.scale(loss).backward()
    else:
        loss.backward()

    return loss


if __name__ == "__main__":
    wandb.login()
    configure_logging()

    parser = argparse.ArgumentParser(description="SimCLR gradients extraction")

    # Standard arguments
    parser.add_argument("--seed", default=42, type=int, help="The seed for random number generators")
    parser.add_argument("--num-workers", type=int, default=18)

    # Model arguments
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--checkpoint", default=None, type=str, help="Path to a model checkpoint. NOTE: we always use pretrained models regardless of whether a checkpoint is specified")
    parser.add_argument("--model", type=ModelType, choices=list(ModelType), default=ModelType.DINO_VIT_16_B, help="The SimCLR backbone (e.g. DINO, CLIP, ...)")
    parser.add_argument("--n-proj-layers", type=int, default=1, help="The number of layers in the projection head. A single projection layer will result in a nonlinearity-free head. If 2+ layers are used they will have nonlinearities in between")
    parser.add_argument("--latent-dim", type=int, default=96, help="The dimensionality of latent representations used to compute the loss")
    parser.add_argument("--use-fp16", action=argparse.BooleanOptionalAction, default=False, help="Whether to run the model in fp16")
    parser.add_argument("--input-size", type=int, default=224, help="Size of the images fed to the model")

    # Dataset arguments
    parser.add_argument("--comparison-batch-dataset", type=DatasetType, choices=list(DatasetType), default=DatasetType.CIFAR10, help="The dataset to use to get the comparison batch")
    parser.add_argument("--comparison-batch-dataset-split", type=DatasetSplit, choices=list(DatasetSplit), required=True, help="The split for the comparison batch dataset")
    parser.add_argument("--comparison-batch-size", type=int, default=256)

    # Hand-picked hard negatives
    parser.add_argument("--hard-negatives-path", type=str, default=None, help="Path to an optional dataset with hard negatives. The number of hard negatives should be smaller than the comparison batch size")

    parser.add_argument("--dataset", type=DatasetType, choices=list(DatasetType), default=DatasetType.CIFAR10, help="The dataset to extract the gradients for")
    parser.add_argument("--dataset-split", type=DatasetSplit, choices=list(DatasetSplit), required=True, help="The dataset split")
    parser.add_argument("--cache-dir", type=str, required=True, help="The cache directory for datasets")

    # Transformation-related arguments
    parser.add_argument("--n-views", type=int, default=2, help="The number of positive views to be generated for each sample")
    parser.add_argument("--temperature", type=float, default=0.07, help="The temperatue for the SimCLR loss")
    parser.add_argument("--augmentation-type", type=SimCLRAugmentationType, choices=list(SimCLRAugmentationType), default=SimCLRAugmentationType.DEFAULT, help="The data augmentation strategy to be used for SimCLR")
    parser.add_argument("--stride-scale", type=int, default=4, help="The stride scale relative to the patch size used in the patchify data augmentation")

    # Hard negatives mixing arguments
    parser.add_argument("--negatives-mixing", action=argparse.BooleanOptionalAction, default=False, help="Whether to add mixed hard negatives to the comparison batch")
    parser.add_argument("--negatives-mixing-k", type=int, default=20, help="Then number of mixed hard negatives to generate")

    # Gradients and output arguments
    parser.add_argument("--fixed-augmentation", action=argparse.BooleanOptionalAction, default=False, help="Whether to always use the same augmentation for positive samples by resetting the seed")
    parser.add_argument("--gradients-layers", type=str, nargs="*", default=None, help="The layers from which gradients will be extracted from. They should be specified as layer_path:dataset_name, where dataset_name is the name of the dataset where the gradients will be saved to")
    parser.add_argument("--output-path", type=str, required=True, help="Where to save the output gradients")
    parser.add_argument("--projection-matrix", type=str, required=True, help="Path to the projection matrix for gradients")

    args = parser.parse_args()

    seed_everything(args.seed)
    main(args)
