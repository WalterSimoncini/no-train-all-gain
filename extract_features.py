import h5py
import torch
import logging
import argparse

from tqdm import tqdm
from torch.utils.data import DataLoader

from src.utils import get_device
from src.models import get_model
from src.datasets import load_dataset
from src.models.rotnet import RotNetType
from src.utils.models import model_feature_dim
from src.utils.logging import configure_logging
from src.enums import ModelType, DatasetType, DatasetSplit


def main(args):
    batch_size = args.batch_size
    model, transform = get_model(
        type_=args.model,
        cache_dir=args.cache_dir,
        n_classes=args.n_classes,
        pretrained=args.pretrained,
        rotnet_type=args.rotnet_type
    )

    ds = load_dataset(
        type_=args.dataset,
        split=args.dataset_split,
        cache_dir=args.cache_dir,
        transform=transform
    )

    logging.info(f"extracting the embeddings for the {args.dataset.value} {args.dataset_split} set...")
    logging.info(f"the data transform is: {transform}")

    ds_loader = DataLoader(
        ds,
        batch_size=batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    device = get_device()

    logging.info(f"loaded model of class {model.__class__.__name__}")
    logging.info(f"the model arguments were (type: {args.model}, pretrained: {args.pretrained}, rotnet: {args.rotnet_type})")
    logging.info(f"the model is {model}")

    # Load the model from a checkpoint if specified
    if args.checkpoint:
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))

    # If we are working with a RotNet remove the heads
    if args.model == ModelType.ROT_NET:
        model = model.model

    model = model.to(device)
    model.eval()

    # Create the output file
    embeddings_file = h5py.File(args.output_file, "w")

    # Create two datasets, one for the embeddings and one for targets
    n_examples = len(ds)
    embeddings_dim = model_feature_dim(model=model, device=device, image_size=args.input_size)

    targets_dataset = embeddings_file.create_dataset("targets", n_examples, dtype="uint32")
    embeddings_dataset = embeddings_file.create_dataset("embeddings", (n_examples, embeddings_dim), dtype="float32")

    with torch.no_grad():
        for i, (images, targets) in tqdm(enumerate(ds_loader)):
            images = images.to(device)
            embeddings = model(images)

            targets_dataset[batch_size * i:batch_size * (i + 1)] = targets.squeeze().numpy()
            embeddings_dataset[batch_size * i:batch_size * (i + 1), :] = embeddings.cpu().numpy()

    embeddings_file.close()


if __name__ == "__main__":
    configure_logging()

    parser = argparse.ArgumentParser(description="Extract features/embeddings for a given model and dataset combination")

    parser.add_argument("--output-file", required=True, type=str, help="Path to the output file")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=18)
    parser.add_argument("--input-size", type=int, default=224, help="Size of the images fed to the model")

    # See the available models here (for DINO):
    # https://github.com/facebookresearch/dino/tree/main#pretrained-models-on-pytorch-hub
    parser.add_argument("--model", type=ModelType, choices=list(ModelType), required=True, help="The type of model to use as a feature extractor")
    parser.add_argument("--checkpoint", default=None, type=str, help="Path to a model checkpoint. If not specified the raw model will be loaded")

    # RotNet arguments (only relevant if model = "rotnet")
    parser.add_argument("--n-classes", default=10, type=int, help="Number of classes in the dataset")
    parser.add_argument("--rotnet-type", type=RotNetType, choices=list(RotNetType), default=None, help="The RotNet type if extracting deep features from a RotNet")
    parser.add_argument("--pretrained", action=argparse.BooleanOptionalAction, default=False, help="Whether to load a pretrained feature extractor or not (only valid for some models)")

    parser.add_argument("--dataset", type=DatasetType, choices=list(DatasetType), default=DatasetType.CIFAR10, help="The dataset to extract features from")
    parser.add_argument("--dataset-split", type=DatasetSplit, choices=list(DatasetSplit), required=True, help="The dataset split")
    parser.add_argument("--cache-dir", type=str, required=True, help="The cache directory for datasets")

    main(parser.parse_args())
