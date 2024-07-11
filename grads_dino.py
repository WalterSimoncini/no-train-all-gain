import h5py
import wandb
import torch
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
from src.utils.dino.transforms import DINODataAgumentation
from src.utils.models import (
    freeze_model,
    model_feature_dim,
    freeze_batchnorm_modules
)


class DINOHead(nn.Module):
    def __init__(self, backbone: nn.Module, embeddings_dim: int, output_dim: int):
        super().__init__()

        self.backbone = backbone
        self.projection = nn.Linear(embeddings_dim, output_dim)

    def forward(self, x):
        x = self.backbone(x)
        x = nn.functional.normalize(x, dim=-1, p=2)

        return self.projection(x)


def main(args):
    wandb.init(project="knnfun", entity="walter-simoncini")

    transform = DINODataAgumentation(crops_size=args.input_size)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    student_backbone, _ = get_model(type_=args.model, cache_dir=args.cache_dir, pretrained=True)
    teacher_backbone, _ = get_model(type_=args.model, cache_dir=args.cache_dir, pretrained=True)

    # Find out the representation dimensionality of the loaded backbone
    student_backbone = student_backbone.to(device)

    embeddings_dim = model_feature_dim(
        student_backbone,
        device=device,
        image_size=args.input_size
    )

    student = DINOHead(
        backbone=student_backbone,
        embeddings_dim=embeddings_dim,
        output_dim=args.latent_dim
    )

    teacher = DINOHead(
        backbone=teacher_backbone,
        embeddings_dim=embeddings_dim,
        output_dim=args.latent_dim
    )

    logging.info(f"the data transform is: {transform}")
    logging.info(f"fixed data augmentation: {args.fixed_augmentation}")

    student, teacher = student.to(device), teacher.to(device)

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

    # The teacher model must be completely frozen
    freeze_model(teacher)

    if args.gradients_layers is not None:
        gradients_layers = parse_gradient_targets(targets=args.gradients_layers)

        # Freeze all modules except the ones we want to extract gradients from
        freeze_model(student, exclusions=list(gradients_layers.keys()))
    else:
        raise ValueError(f"You must specify at least one mapping in --gradients-layers")

    logging.info(f"freezing batch norm modules...")

    freeze_batchnorm_modules(student, device=device)
    freeze_batchnorm_modules(teacher, device=device)

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
    autograd_hacks.add_hooks(student, layer_paths=gradient_paths)

    for batch_index, (views, _) in tqdm(enumerate(data_loader)):
        # Batch size and number of crops
        batch_offset = args.batch_size * batch_index
        actual_batch_size, num_crops = views.shape[:2]

        step(
            student=student,
            teacher=teacher,
            views=views,
            device=device,
            use_fp16=args.use_fp16,
            scaler=scaler,
            teacher_temperature=args.teacher_temperature,
            student_temperature=args.student_temperature
        )

        # Compute the per-sample gradients
        autograd_hacks.compute_grad1(student, layer_paths=gradient_paths)

        # Log the current step to wandb so we can also observe
        # some system metrics (e.g. GPU memory and resource
        # utilization)
        wandb.log({ "step": batch_index })

        for layer_path, dataset_name in gradients_layers.items():
            layer = get_layer(model=student, path=layer_path)

            # This code works regardless of the batch size (even for batch size = 1)
            if type(layer) == nn.Conv2d:
                batch_gradients = layer.weight.grad1.reshape(actual_batch_size, num_crops, -1).sum(dim=1)
            else:
                H, W = layer.weight.shape[:2]

                weight_gradient = layer.weight.grad1.reshape(actual_batch_size, num_crops, H, W)
                weight_gradient = weight_gradient.sum(dim=1)

                if hasattr(layer, "bias") and layer.bias is not None:
                    # Extract the gradient for the bias vector as well
                    bias_gradient = layer.bias.grad1.sum(dim=1).reshape(actual_batch_size, num_crops, -1)
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
    student: nn.Module,
    teacher: nn.Module,
    views: torch.Tensor,
    device: torch.device,
    use_fp16: bool = False,
    scaler: torch.cuda.amp.GradScaler = None,
    teacher_temperature: float = 0.07,
    student_temperature: float = 0.1,
    num_global_crops: int = 2,
    skip_same_view: bool = False
):
    student.zero_grad()

    # The views have shape [batch, crops, C, H, W]
    B, CR, C, H, W = views.shape

    views = views.to(device)

    # Select the global views (i.e. the first two crops)
    global_views = views[:, :2].reshape(-1, C, H, W)
    views = views.reshape(-1, C, H, W)

    # Clear tensors used for the per-sample gradient computation
    autograd_hacks.clear_backprops(student)

    with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_fp16):
        # The teacher and student views must go through two different model instances
        student_embeddings = student(views)
        teacher_embeddings = teacher(global_views).detach()

        student_emb = student_embeddings.reshape(B, CR, -1)
        teacher_emb = teacher_embeddings.reshape(B, num_global_crops, -1)

        # Calculate the DINO loss
        teacher_emb = nn.functional.softmax(teacher_emb / teacher_temperature, dim=-1)
        student_emb = nn.functional.log_softmax(student_emb / student_temperature, dim=-1)

        # Convert the teacher and student embeddings in [CR, B, E]
        teacher_emb = teacher_emb.permute(1, 0, 2)
        student_emb = student_emb.permute(1, 0, 2)

        losses = torch.zeros(B).to(device)

        for ti, tv in enumerate(teacher_emb):
            for si, sv in enumerate(student_emb):
                if skip_same_view and ti == si:
                    # Skip cases where teacher and student operate on the same view
                    continue

                losses += (-tv * sv).sum(dim=-1)

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
    parser.add_argument("--latent-dim", type=int, default=65536, help="The dimensionality of latent representations used to compute the loss")
    parser.add_argument("--use-fp16", action=argparse.BooleanOptionalAction, default=False, help="Whether to run the model in fp16")
    parser.add_argument("--model", type=ModelType, choices=list(ModelType), required=True, help="The type of model to use as a feature extractor")
    parser.add_argument("--input-size", type=int, default=224, help="Size of the images fed to the model")

    # Dataset arguments
    parser.add_argument("--dataset", type=DatasetType, choices=list(DatasetType), default=DatasetType.CIFAR10, help="The dataset to extract the gradients for")
    parser.add_argument("--dataset-split", type=DatasetSplit, choices=list(DatasetSplit), required=True, help="The dataset split")
    parser.add_argument("--cache-dir", type=str, required=True, help="The cache directory for datasets")

    # Transformation-related arguments
    parser.add_argument("--teacher-temperature", type=float, default=0.07, help="The temperatue for the teacher latents")
    parser.add_argument("--student-temperature", type=float, default=0.1, help="The temperatue for the student latents")

    # Gradients and output arguments
    parser.add_argument("--fixed-augmentation", action=argparse.BooleanOptionalAction, default=False, help="Whether to always use the same augmentation for positive samples by resetting the seed")
    parser.add_argument("--gradients-layers", type=str, nargs="*", default=None, help="The layers from which gradients will be extracted from. They should be specified as layer_path:dataset_name, where dataset_name is the name of the dataset where the gradients will be saved to")
    parser.add_argument("--output-path", type=str, required=True, help="Where to save the output gradients")
    parser.add_argument("--projection-matrix", type=str, required=True, help="Path to the projection matrix for gradients")

    args = parser.parse_args()

    seed_everything(args.seed)
    main(args)
