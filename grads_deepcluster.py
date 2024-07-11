import os
import h5py
import wandb
import torch
import logging
import argparse
import numpy as np
import torch.nn as nn
import src.utils.autograd_hacks as autograd_hacks

from tqdm import tqdm
from typing import List, Tuple, Dict
from torch.utils.data import DataLoader
from torch.nn.functional import cross_entropy

from src.models import get_model
from scipy.sparse import csr_matrix
from src.utils.grads import get_layer
from src.datasets import load_dataset
from src.utils.seed import seed_everything
from src.utils.logging import configure_logging
from src.utils.args import parse_gradient_targets
from src.enums import DatasetSplit, ModelType, DatasetType
from src.utils.deepcluster.dataset import MultiCropDataset
from src.utils.models import (
    freeze_model,
    model_feature_dim,
    freeze_batchnorm_modules
)


class DeepClusterHead(nn.Module):
    def __init__(self, backbone: nn.Module, embeddings_dim: int, latent_dim: int, num_prototypes: List[int]):
        """
            :param num_prototypes: list containing the number of clusters for each head
        """
        super().__init__()

        self.backbone = backbone

        # We use weight normalization for the projection layer following
        # the original DINO repository
        self.projection = nn.utils.weight_norm(nn.Linear(embeddings_dim, latent_dim, bias=False))
        self.projection.weight_g.data.fill_(1)

        self.latent_dim = latent_dim
        # self.projection = nn.Linear(embeddings_dim, latent_dim)
        self.num_heads = len(num_prototypes)

        for i, k in enumerate(num_prototypes):
            self.add_module(f"prototypes_{i}", nn.Linear(latent_dim, k, bias=False))

    def forward(self, x):
        # FIXME: They also use a multi-crop wrapper, maybe that has an effect
        x = self.backbone(x)
        x = self.projection(x)

        # Originally dim=1, but this should still work
        x = nn.functional.normalize(x, dim=-1, p=2)

        prototypes = [
            getattr(self, f"prototypes_{i}")(x) for i in range(self.num_heads)
        ]

        return x, prototypes


def create_memory_bank(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    crops_for_assignment: List[int] = [0, 1]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
        :param crops_for_assignment: the indices of the crops used to build the
                                     memory bank. The global crops are used by default
    """
    num_examples = len(data_loader.dataset)
    crops_for_assignment = torch.tensor(crops_for_assignment, dtype=torch.int64)

    # Create a [crops_for_assignment, dataset] sized memory and an indices
    # array to map mempry embeddings to their original dataset samples
    memory_indices = torch.zeros(num_examples, dtype=torch.int64)
    memory_embeddings = torch.zeros(
        num_examples,
        len(crops_for_assignment),
        model.latent_dim
    )

    logging.info(f"creating a memory bank of size {memory_embeddings.shape}")

    with torch.no_grad():
        for batch_index, (images, indices) in enumerate(data_loader):
            # images is a tensor of size [B, CR, C, H, W]
            # indices is a tensor of size [B]
            actual_batch_size = images.shape[0]
            batch_offset = batch_index * data_loader.batch_size

            indices = indices.to(device)

            # Only select the crops used to build the memory bank
            images = images[:, crops_for_assignment, :, :, :]
            images = images.to(device, non_blocking=True)

            # Reshape the crops to form a single batch
            B, CR, C, H, W = images.shape

            images = images.reshape(B * CR, C, H, W)

            # Compute the image embeddings and reshape them
            # to be [batch, crop, embedding]
            embeddings, _ = model(images)
            embeddings = embeddings.reshape(B, CR, -1)

            # Store the indices and crop embeddings of the current batch
            memory_indices[batch_offset:batch_offset + actual_batch_size] = indices
            memory_embeddings[batch_offset:batch_offset + actual_batch_size, :, :] = embeddings

    return memory_embeddings, memory_indices


def cluster_memory_bank(
    model: nn.Module,
    memory_embeddings: torch.Tensor,
    memory_indices: torch.Tensor,
    dataset_size: int,
    device: torch.device,
    num_prototypes: List[int],
    num_iters: int = 10,
) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """
        Runs k-means clustering over the memory bank embeddings and returns
        the centroid assignments of each memory bank item

        :param num_heads: number of centroids heads for the model
    """
    logging.info("computing the cluster centroids and assignments")

    # This array has generally a size of [dataset, 3].
    # the memory embeddings have shape [dataset, crops, embedding]
    memory_indices = memory_indices.to(device)
    memory_embeddings = memory_embeddings.to(device)

    current_crop_index = 0
    assignments = -100 * torch.ones(
        dataset_size,
        len(num_prototypes),
        dtype=torch.int64
    ).to(device)

    with torch.no_grad():
        centroids_bank = {}

        for j, num_centroids in enumerate(num_prototypes):
            assert num_centroids <= dataset_size, f"the number of centroids {num_centroids} must be smaller than the dataset size {dataset_size}"

            # Initialize the centroids using alternated crops in the memory bank
            centroid_indices = torch.randperm(dataset_size)[:num_centroids]
            centroids = memory_embeddings[centroid_indices, current_crop_index, :]

            for i in range(num_iters + 1):
                dot_products = torch.mm(
                    memory_embeddings[:, current_crop_index, :],
                    centroids.t()
                )

                _, local_assignments = dot_products.max(dim=1)

                if i == num_iters:
                    # Terminate the clustering when the max number of iteration
                    # has been reached after computing the last assignemnts
                    break

                # Copied as-is from the original code. Not exactly sure
                # what is happening here on a line-by-line basis
                where_helper = get_indices_sparse(local_assignments.cpu().numpy())

                counts = torch.zeros(num_centroids).to(device, non_blocking=True).int()
                emb_sums = torch.zeros(num_centroids, model.latent_dim).to(device, non_blocking=True)

                for k in range(len(where_helper)):
                    if len(where_helper[k][0]) > 0:
                        emb_sums[k] = torch.sum(
                            memory_embeddings[:, current_crop_index, :][where_helper[k][0]],
                            dim=0
                        )

                        counts[k] = len(where_helper[k][0])

                mask = counts > 0

                centroids[mask] = emb_sums[mask] / counts[mask].unsqueeze(1)

            centroids_bank[j] = centroids.clone()

            # Assign the centroids to their corresponding model head
            getattr(model, f"prototypes_{j}").weight.copy_(centroids)

            # Record assignments for this head
            assignments[memory_indices, j] = local_assignments

            # Cycle through the crops in the memory bank
            current_crop_index = (current_crop_index + 1) % memory_embeddings.shape[1]

    return assignments, centroids_bank


def assign_memory_bank(
    model: nn.Module,
    memory_embeddings: torch.Tensor,
    memory_indices: torch.Tensor,
    dataset_size: int,
    device: torch.device,
    num_prototypes: List[int],
    centroids_bank: Dict[int, torch.Tensor]
) -> torch.Tensor:
    logging.info("assigning the memory bank to precomputed centroids")

    memory_embeddings = memory_embeddings.to(device)

    current_crop_index = 0
    assignments = -100 * torch.ones(
        dataset_size,
        len(num_prototypes),
        dtype=torch.int64
    ).to(device)

    with torch.no_grad():
        for j, _ in enumerate(num_prototypes):
            centroids = centroids_bank[j].to(device)

            dot_products = torch.mm(
                memory_embeddings[:, current_crop_index, :],
                centroids.t()
            )

            _, local_assignments = dot_products.max(dim=1)

            # Assign the centroids to their corresponding model head
            getattr(model, f"prototypes_{j}").weight.copy_(centroids)

            # Record assignments for this head
            assignments[memory_indices, j] = local_assignments

            # Cycle through the crops in the memory bank
            current_crop_index = (current_crop_index + 1) % memory_embeddings.shape[1]

    return assignments


def get_indices_sparse(data):
    cols = np.arange(data.size)

    M = csr_matrix(
        (cols, (data.ravel(), cols)), shape=(int(data.max()) + 1, data.size)
    )

    return [np.unravel_index(row.data, data.shape) for row in M]


def main(args):
    wandb.init(project="knnfun", entity="walter-simoncini")

    # Verify that the centroids file exists if we are running
    # the gradient extraction for the test/validation sets
    if not args.is_train:
        assert os.path.isfile(args.clusters_path)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model, _ = get_model(type_=args.model, cache_dir=args.cache_dir, pretrained=True)
    model = model.to(device)

    # Find out the representation dimensionality of the loaded backbone
    embeddings_dim = model_feature_dim(model, device=device)

    model = DeepClusterHead(
        backbone=model,
        embeddings_dim=embeddings_dim,
        latent_dim=args.latent_dim,
        num_prototypes=args.num_prototypes
    ).to(device)

    logging.info(f"the model is {model}")

    base_dataset = load_dataset(
        type_=args.dataset,
        split=args.dataset_split,
        cache_dir=args.cache_dir
    )

    dataset = MultiCropDataset(
        dataset=base_dataset,
        crops_size=args.crops_sizes,
        num_crops=args.num_crops,
        min_scale_crops=args.min_scale_crops,
        max_scale_crops=args.max_scale_crops,
        return_index=True
    )

    data_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True
    )

    logging.info(f"the dataset is {base_dataset} (split: {args.dataset_split})")
    logging.info(f"the data transforms are: {dataset.transform}")
    logging.info(f"fixed data augmentation: {args.fixed_augmentation}")

    if args.gradients_layers is not None:
        gradients_layers = parse_gradient_targets(targets=args.gradients_layers)

        # Freeze all modules except the ones we want to extract gradients from
        freeze_model(model, exclusions=list(gradients_layers.keys()))
    else:
        raise ValueError(f"You must specify at least one mapping in --gradients-layers")

    logging.info(f"freezing batch norm modules...")

    freeze_batchnorm_modules(model, device=device)

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

    # Build the memory bank and perform clustering if necessary
    logging.info("building the memory bank")

    memory_bank, memory_indices = create_memory_bank(
        model=model,
        data_loader=data_loader,
        device=device,
        crops_for_assignment=args.crops_for_assignment
    )

    if args.is_train:
        assignments, centroids_bank = cluster_memory_bank(
            model=model,
            memory_embeddings=memory_bank,
            memory_indices=memory_indices,
            dataset_size=len(data_loader.dataset),
            device=device,
            num_prototypes=args.num_prototypes
        )

        torch.save(centroids_bank, args.clusters_path)
    else:
        assignments = assign_memory_bank(
            model=model,
            memory_embeddings=memory_bank,
            memory_indices=memory_indices,
            dataset_size=len(data_loader.dataset),
            device=device,
            num_prototypes=args.num_prototypes,
            centroids_bank=torch.load(args.clusters_path, map_location=device)
        )

    logging.info("extracting gradients")

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

    total_num_crops = sum(args.num_crops)

    for batch_index, (images, indices) in tqdm(enumerate(data_loader)):
        actual_batch_size = images.shape[0]
        batch_offset = args.batch_size * batch_index

        step(
            model=model,
            images=images,
            indices=indices,
            assignments=assignments,
            device=device,
            use_fp16=args.use_fp16,
            scaler=scaler,
            temperature=args.temperature,
            num_heads=len(args.num_prototypes)
        )

        # Compute the per-sample gradients
        autograd_hacks.compute_grad1(model, layer_paths=gradient_paths)

        # Log the current step to wandb so we can also observe
        # some system metrics (e.g. GPU memory and resource
        # utilization)
        wandb.log({ "step": batch_index })

        for layer_path, dataset_name in gradients_layers.items():
            layer = get_layer(model=model, path=layer_path)

            if type(layer) == nn.Conv2d:
                batch_gradients = layer.weight.grad1.reshape(actual_batch_size, total_num_crops, -1).sum(dim=1)
            else:
                # This code works regardless of the batch size (even for batch size = 1)
                H, W = layer.weight.shape[:2]

                # We should convert gradients to fp16 right away
                weight_gradient = layer.weight.grad1.reshape(actual_batch_size, total_num_crops, H, W)
                weight_gradient = weight_gradient.sum(dim=1)

                if hasattr(layer, "bias") and layer.bias is not None:
                    # Extract the gradient for the bias vector as well
                    bias_gradient = layer.bias.grad1.sum(dim=1).reshape(actual_batch_size, total_num_crops, -1)
                    bias_gradient = bias_gradient.sum(dim=1).unsqueeze(dim=-1)

                    batch_gradients = torch.cat([weight_gradient, bias_gradient], dim=-1)
                else:
                    batch_gradients = weight_gradient

            # Project and save the extracted gradients
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
    images: torch.Tensor,
    indices: torch.Tensor,
    assignments: torch.Tensor,
    device: torch.device,
    use_fp16: bool = True,
    scaler: torch.cuda.amp.GradScaler = None,
    temperature: float = 0.1,
    num_heads: int = 3
):
    model.zero_grad()

    # Clear tensors used for the per-sample gradient computation
    autograd_hacks.clear_backprops(model)

    # Reshape the crops to form a single batch
    B, CR, C, H, W = images.shape

    images = images.reshape(-1, C, H, W).to(device)

    with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_fp16):
        # The assignments should have shape [3, B, 3000] and be a list
        _, model_scores = model(images)

        # [B, CR, H, P], where B is the batch, CR the number of crops,
        # H the number of heads and P the number of prototypes
        model_scores = torch.cat([x.unsqueeze(dim=0) for x in model_scores])
        model_scores = model_scores.permute(1, 0, 2).reshape(B, CR, num_heads, -1)
        model_scores = model_scores / temperature

        losses = torch.zeros(B).to(device)
        num_prototypes = model_scores.shape[-1]

        # Iterate over batch items and calculate a loss for each of them
        for i, (scores, assignment_index) in enumerate(zip(model_scores, indices)):
            # Scores is [H, CR, P], targets is [H, B]
            scores = scores.permute(1, 0, 2)
            # Assignments is [D, H], where D is the dataset
            # size and H the number of heads
            targets = assignments[assignment_index].unsqueeze(dim=0).permute(1, 0).repeat(1, CR)

            scores = scores.reshape(-1, num_prototypes)
            targets = targets.reshape(-1)

            losses[i] = cross_entropy(scores, targets, ignore_index=-100)

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
    parser.add_argument("--latent-dim", type=int, default=128, help="The dimensionality of latent representations used to compute the cluster assignments")
    parser.add_argument("--use-fp16", action=argparse.BooleanOptionalAction, default=False, help="Whether to run the model in fp16")
    parser.add_argument("--model", type=ModelType, choices=list(ModelType), required=True, help="The type of model to use as a feature extractor")
 
    # Dataset arguments
    parser.add_argument("--dataset", type=DatasetType, choices=list(DatasetType), default=DatasetType.CIFAR10, help="The dataset to extract the gradients for")
    parser.add_argument("--dataset-split", type=DatasetSplit, choices=list(DatasetSplit), required=True, help="The dataset split")
    parser.add_argument("--cache-dir", type=str, required=True, help="The cache directory for datasets")

    # Transformation-related arguments
    parser.add_argument("--num-crops", type=int, nargs="*", default=[2, 6], help="The number of global and local crops")
    parser.add_argument("--crops-sizes", type=int, nargs="*", default=[224, 224], help="The width/height of global and local crops")
    parser.add_argument("--crops-for-assignment", type=int, nargs="*", default=[0, 1], help="Indices of the crops used to calculate the cluster centroids")
    parser.add_argument("--min-scale-crops", type=float, nargs="*", default=[0.14, 0.05], help="The minimum scale of global and local crops")
    parser.add_argument("--max-scale-crops", type=float, nargs="*", default=[1.0, 0.14], help="The maximum scale of global and local crops")

    # Gradients and output arguments
    parser.add_argument("--fixed-augmentation", action=argparse.BooleanOptionalAction, default=False, help="Whether to always use the same augmentation for positive samples by resetting the seed")
    parser.add_argument("--gradients-layers", type=str, nargs="*", default=None, help="The layers from which gradients will be extracted from. They should be specified as layer_path:dataset_name, where dataset_name is the name of the dataset where the gradients will be saved to")
    parser.add_argument("--output-path", type=str, required=True, help="Where to save the output gradients")
    parser.add_argument("--projection-matrix", type=str, required=True, help="Path to the projection matrix for gradients")

    # Deepcluster-specific arguments
    parser.add_argument("--is-train", action=argparse.BooleanOptionalAction, default=False, help="Whether we are extracting gradients for the training dataset. Causes clusters to be recomputed")
    parser.add_argument("--temperature", type=float, default=0.1, help="The loss temperature")
    parser.add_argument("--num-prototypes", type=int, nargs="*", default=[3000, 3000, 3000], help="The number of centroids/prototypes for each model head")
    parser.add_argument("--clusters-path", type=str, required=True, help="Where to save/load centroids from")

    args = parser.parse_args()

    seed_everything(args.seed)
    main(args)
