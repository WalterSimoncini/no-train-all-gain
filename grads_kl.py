import h5py
import torch
import logging
import argparse
import torch.nn as nn
import src.utils.autograd_hacks as autograd_hacks

from tqdm import tqdm
from torch.utils.data import DataLoader

from src.utils import get_device
from src.models import get_model
from src.utils.grads import get_layer
from src.datasets import load_dataset
from src.utils.models import freeze_model
from src.utils.seed import seed_everything
from torch.nn.functional import log_softmax, softmax, kl_div
from src.utils.logging import configure_logging
from src.utils.args import parse_gradient_targets
from src.enums import DatasetSplit, ModelType, DatasetType
from src.utils.models import (
    model_feature_dim,
    freeze_batchnorm_modules
)


class KLHead(nn.Module):
    def __init__(self, backbone: nn.Module, embeddings_dim: int, output_dim: int):
        super().__init__()

        self.backbone = backbone
        self.projection = nn.Linear(embeddings_dim, output_dim)

    def forward(self, x):
        x = self.backbone(x)
        x = nn.functional.normalize(x, dim=-1, p=2)

        return self.projection(x)


def main(args):
    logging.info("started the supervised gradients extraction")
    logging.info(f"the script arguments were: {args}")

    model, transform = get_model(
        type_=args.model,
        cache_dir=args.cache_dir,
        pretrained=True
    )

    device = get_device()
    model = model.to(device)

    embedding_size = model_feature_dim(
        model=model,
        device=device,
        image_size=args.input_size
    )

    model = KLHead(
        backbone=model,
        embeddings_dim=embedding_size,
        output_dim=args.latent_dim
    ).to(device)

    logging.info(f"the data transform is: {transform}")
    logging.info(f"the model is {type(model.backbone)} (the model argument was: {args.model})")

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

        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    else:
        logging.info("no checkpoint was specified: loading a raw model")

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

    uniform = (torch.ones(args.latent_dim) / args.latent_dim).to(device)
    softmax_uniform = softmax(uniform / args.temperature, dim=0)
    softmax_uniform = softmax_uniform.unsqueeze(dim=0).repeat(args.batch_size, 1)

    scaler = torch.cuda.amp.GradScaler(enabled=args.use_fp16)
    gradient_paths = autograd_hacks.preprocess_layer_paths(
        paths=list(gradients_layers.keys())
    )

    # Add hooks to calculate the per-sample gradients
    autograd_hacks.add_hooks(model, layer_paths=gradient_paths)

    for batch_index, (images, _) in tqdm(enumerate(data_loader)):
        actual_batch_size = images.shape[0]
        batch_offset = args.batch_size * batch_index

        step(
            model=model,
            images=images,
            device=device,
            softmax_uniform=softmax_uniform[:actual_batch_size, :],
            use_fp16=args.use_fp16,
            scaler=scaler,
            temperature=args.temperature
        )

        # Compute the per-sample gradients
        autograd_hacks.compute_grad1(model, layer_paths=gradient_paths)

        # Project and write gradients to disk
        for layer_path, dataset_name in gradients_layers.items():
            layer = get_layer(model=model, path=layer_path)

            if type(layer) == nn.Conv2d:
                batch_gradients = layer.weight.grad1
            else:
                if hasattr(layer, "bias") and layer.bias is not None:
                    batch_gradients = torch.cat([
                        layer.weight.grad1,
                        layer.bias.grad1.sum(dim=1).unsqueeze(dim=-1)
                    ], dim=-1)
                else:
                    batch_gradients = layer.weight.grad1

            if args.use_fp16:
                batch_gradients = batch_gradients.to(torch.bfloat16)

            batch_gradients = batch_gradients.view(actual_batch_size, -1)
            batch_gradients = scaling * (projection @ batch_gradients.T).permute(1, 0)

            grads_file[dataset_name][batch_offset:batch_offset + actual_batch_size] = batch_gradients.to(torch.float32).cpu().numpy()

    grads_file.close()

    logging.info(f"saved the gradients to {args.output_path}")


def step(
    model: nn.Module,
    images: torch.Tensor,
    device: torch.device,
    softmax_uniform: torch.Tensor,
    use_fp16: bool = False,
    scaler: torch.cuda.amp.GradScaler = None,
    temperature: float = 15
) -> torch.Tensor:
    model.zero_grad()

    # Clear tensors used for the per-sample gradient computation
    autograd_hacks.clear_backprops(model)

    images = images.to(device)

    with torch.cuda.amp.autocast(dtype=torch.bfloat16, enabled=use_fp16):
        preds = model(images)

        softmax_preds = log_softmax(preds / temperature, dim=1)
        loss = kl_div(softmax_preds, softmax_uniform, reduction="mean")

        assert not torch.isnan(loss), "found a NaN loss. Stopping the gradient extraction"

    if use_fp16:
        scaler.scale(loss).backward()
    else:
        loss.backward()

    return loss


if __name__ == "__main__":
    configure_logging()

    parser = argparse.ArgumentParser(description="Supervised (KL) gradients extraction")

    # Standard arguments
    parser.add_argument("--seed", default=42, type=int, help="The seed for random number generators")
    parser.add_argument("--num-workers", type=int, default=18)

    # Model arguments
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--checkpoint", default=None, type=str, help="Path to the RotNet checkpoint. If not specified a raw model will be loaded")
    parser.add_argument("--model", type=ModelType, choices=list(ModelType), default=ModelType.VIT_B_16, help="The type of the supervised backbone")
    parser.add_argument("--use-fp16", action=argparse.BooleanOptionalAction, default=False, help="Whether to run the model in fp16")
    parser.add_argument("--latent-dim", type=int, default=768, help="The output dimensionality of the KL head")
    parser.add_argument("--input-size", type=int, default=224, help="Size of the images fed to the model")

    # Dataset arguments
    parser.add_argument("--dataset", type=DatasetType, choices=list(DatasetType), default=DatasetType.CIFAR10, help="The dataset to extract the gradients for")
    parser.add_argument("--dataset-split", type=DatasetSplit, choices=list(DatasetSplit), required=True, help="The dataset split")
    parser.add_argument("--cache-dir", type=str, required=True, help="The cache directory for datasets")

    # Gradients and output arguments
    parser.add_argument("--temperature", type=float, default=1.0, help="The softmax temperature for the loss")
    parser.add_argument("--gradients-layers", type=str, nargs="*", default=None, help="The layers from which gradients will be extracted from. They should be specified as layer_path:dataset_name, where dataset_name is the name of the dataset where the gradients will be saved to")
    parser.add_argument("--output-path", type=str, required=True, help="Where to save the output gradients")
    parser.add_argument("--projection-matrix", type=str, required=True, help="Path to the projection matrix for gradients")

    args = parser.parse_args()

    seed_everything(args.seed)
    main(args)
