import h5py
import torch
import logging
import argparse
import torch.nn as nn
import src.utils.autograd_hacks as autograd_hacks

from tqdm import tqdm
from itertools import chain
from torch.utils.data import DataLoader

from src.utils.simclr import info_nce_loss
from src.models import get_model, ModelType
from src.utils.transforms import ContrastiveRandomCrop
from src.datasets import load_dataset, DatasetSplit, DatasetType
from src.utils.misc import seed_everything, configure_logging, parse_gradient_targets
from src.utils.grads import get_layer, freeze_model, freeze_batchnorm_modules


class SimCLRHead(nn.Module):
    def __init__(self, backbone: nn.Module, embeddings_dim: int, output_dim: int):
        super().__init__()

        self.backbone = backbone
        self.projection = nn.Linear(embeddings_dim, output_dim)

    def forward(self, **kwargs):
        # Only select the CLS token
        x = self.backbone(**kwargs).last_hidden_state[:, 0, :]

        return self.projection(x)


def main(args):
    logging.info("started the supervised gradients extraction")
    logging.info(f"the script arguments were: {args}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, tokenizer = get_model(type_=args.model)
    model = model.to(device)

    dataset = load_dataset(
        type_=args.dataset,
        split=args.dataset_split
    )

    logging.info(f"extracting gradients for {dataset} using model {args.model}")

    model = SimCLRHead(
        backbone=model,
        embeddings_dim=args.embeddings_dim,
        output_dim=args.latent_dim
    ).to(device)

    transform = ContrastiveRandomCrop(num_views=args.num_views)
    comparison_batch = precompute_comparison_batch(
        model=model,
        tokenizer=tokenizer,
        device=device,
        dataset_name=args.dataset,
        transform=ContrastiveRandomCrop(num_views=2),
        num_negatives=args.comparison_batch_size
    )

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    logging.info(f"freezing batch norm modules...")
    freeze_batchnorm_modules(model, device=device)

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

    # Create a file/dataset to store the computed gradients
    grads_file = h5py.File(args.output_path, "w")

    for layer_path, dataset_name in gradients_layers.items():
        grads_file.create_dataset(dataset_name, (len(data_loader.dataset), feature_dim), dtype="float32")

    # Calculate the supervised loss for each sample, backpropagate
    # and extract the gradients from the target layers. In this
    # case the "supervised" loss is the KL(softmax||U), as the
    # cross-entropy loss would lead to a 1.0 accuracy, as it
    # encodes the target class directly
    logging.info(f"extracting gradients...")
    logging.info(f"the features have dimension {args.latent_dim}")

    scaler = torch.cuda.amp.GradScaler(enabled=args.use_fp16)
    gradient_paths = autograd_hacks.preprocess_layer_paths(
        paths=list(gradients_layers.keys())
    )

    # Add hooks to calculate the per-sample gradients
    autograd_hacks.add_hooks(model, layer_paths=gradient_paths)

    for batch_index, (sentences, _) in tqdm(enumerate(data_loader)):
        actual_batch_size = len(sentences)
        batch_offset = args.batch_size * batch_index

        model.zero_grad()

        # Clear tensors used for the per-sample gradient computation
        autograd_hacks.clear_backprops(model)

        with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=args.use_fp16):
            # Now sentences is an array of size B where every element
            # is an array of size V, i.e. the number of views per sentence
            sentences = transform(texts=sentences)

            # Flatten the array, so now we have an array of size B * V
            sentences = chain.from_iterable(sentences)
            sentences = list(sentences)

            # Tokenize and encode sentences
            tokens = tokenizer(sentences, return_tensors="pt", padding=True).to(device)
            preds = model(**tokens)

            # Calculate the contrastive loss
            losses = torch.zeros(len(sentences), device=device)

            for i in range(actual_batch_size):
                # Select the positive views corresponding to this batch item
                positive_views = preds[args.num_views * i:args.num_views * (i + 1)]

                losses[i] = info_nce_loss(
                    comparison_batch_features=comparison_batch,
                    positive_features=positive_views
                )

            loss = losses.mean()

            assert not torch.isnan(loss), "found a NaN loss. Stopping the gradient extraction"

        if args.use_fp16:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        # Compute the per-sample gradients
        autograd_hacks.compute_grad1(model, layer_paths=gradient_paths)

        # Project and write gradients to disk
        for layer_path, dataset_name in gradients_layers.items():
            layer = get_layer(model=model, path=layer_path)

            _, _, W, H = layer.weight.grad1.shape

            if type(layer) == nn.Conv2d:
                batch_gradients = layer.weight.grad1
            else:
                if hasattr(layer, "bias") and layer.bias is not None:
                    batch_gradients = torch.cat([
                        # Only the CLS (0) token has non-zero gradients
                        layer.weight.grad1[:, 0, :, :].reshape(actual_batch_size, args.num_views, W, H).sum(dim=1),
                        layer.bias.grad1[:, 0, :].reshape(actual_batch_size, args.num_views, -1).sum(dim=1).unsqueeze(dim=-1)
                    ], dim=-1)
                else:
                    batch_gradients = layer.weight.grad1[:, 0, :, :].reshape(
                        actual_batch_size,
                        args.num_views,
                        W,
                        H
                    ).sum(dim=1)

            if args.use_fp16:
                batch_gradients = batch_gradients.to(torch.bfloat16)

            batch_gradients = batch_gradients.view(actual_batch_size, -1)
            batch_gradients = scaling * (projection @ batch_gradients.T).permute(1, 0)

            grads_file[dataset_name][batch_offset:batch_offset + actual_batch_size] = batch_gradients.to(torch.float32).cpu().numpy()

    grads_file.close()

    logging.info(f"saved the gradients to {args.output_path}")


def precompute_comparison_batch(
    model: nn.Module,
    tokenizer: nn.Module,
    device: torch.device,
    dataset_name: str,
    transform: nn.Module,
    num_negatives: int = 128
) -> torch.Tensor:
    train_dataset = load_dataset(type_=dataset_name, split=DatasetSplit.TRAIN)

    negatives_batch = []
    negative_indices = torch.randperm(len(train_dataset))[:num_negatives]

    with torch.no_grad():
        for index in negative_indices:
            sample, _ = train_dataset[index]
            sentence_views = transform(texts=[sample]).pop()

            tokens = tokenizer(sentence_views, return_tensors="pt", padding=True).to(device)
            preds = model(**tokens)

            negatives_batch.append(preds)

    return torch.cat(negatives_batch, dim=0)


if __name__ == "__main__":
    configure_logging()

    parser = argparse.ArgumentParser(description="Supervised (KL) gradients extraction")

    # Standard arguments
    parser.add_argument("--seed", default=42, type=int, help="The seed for random number generators")
    parser.add_argument("--num-workers", type=int, default=18)

    # Model arguments
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--use-fp16", action=argparse.BooleanOptionalAction, default=False, help="Whether to run the model in fp16")
    parser.add_argument("--latent-dim", type=int, default=768, help="The output dimensionality of the KL head")
    parser.add_argument("--model", type=ModelType, choices=list(ModelType), default=ModelType.BERT_BASE_UNCASED, help="The model used to extract features")
    parser.add_argument("--embeddings-dim", type=int, default=768, help="The model output dimensionality")

    # Dataset arguments
    parser.add_argument("--dataset", type=DatasetType, choices=list(DatasetType), default=DatasetType.FINE_GRAINED_SST, help="The dataset to extract features from")
    parser.add_argument("--dataset-split", type=DatasetSplit, choices=list(DatasetSplit), required=True, help="The dataset split")

    # Loss arguments
    parser.add_argument("--comparison-batch-size", type=int, default=128, help="Number of negatives for the comparison batch")
    parser.add_argument("--num-views", type=int, default=4, help="Number of views generated for each positive/negative sample")

    # Gradients and output arguments
    parser.add_argument("--gradients-layers", type=str, nargs="*", default=None, help="The layers from which gradients will be extracted from. They should be specified as layer_path:dataset_name, where dataset_name is the name of the dataset where the gradients will be saved to")
    parser.add_argument("--output-path", type=str, required=True, help="Where to save the output gradients")
    parser.add_argument("--projection-matrix", type=str, required=True, help="Path to the projection matrix for gradients")

    args = parser.parse_args()

    seed_everything(args.seed)
    main(args)
