import h5py
import torch
import logging
import argparse

from tqdm import tqdm
from torch.utils.data import DataLoader

from src.fungi.models import get_model
from src.fungi.datasets import load_dataset
from src.fungi.utils.misc import configure_logging
from src.fungi.enums import ModelType, DatasetType, DatasetSplit


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = load_dataset(
        type_=args.dataset,
        split=args.dataset_split,
        cache_dir=args.cache_dir
    )

    model = get_model(
        type_=args.model,
        cache_dir=args.cache_dir,
        input_tdim=dataset.audio_conf["target_length"]
    ).to(device)

    model = model.eval()

    logging.info(f"extracting embeddings for {dataset} using model {args.model}")

    batch_size = args.batch_size
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False
    )

    # Create the output file
    embeddings_file = h5py.File(args.output_path, "w")

    # Create two datasets, one for the embeddings and one for targets
    n_examples = len(dataset)

    targets_dataset = embeddings_file.create_dataset("targets", n_examples, dtype="uint32")
    embeddings_dataset = embeddings_file.create_dataset("embeddings", (n_examples, args.embeddings_dim), dtype="float32")

    with torch.no_grad():
        for i, (samples, targets) in tqdm(enumerate(data_loader)):
            # The targets are one-hot encoded, so to get the actual target
            # value we get its argmax, which corresponds to the label index
            targets = targets.argmax(dim=1)

            samples = samples.to(device)
            embeddings = model(samples, task="ft_cls")

            targets_dataset[batch_size * i:batch_size * (i + 1)] = targets.squeeze().numpy()
            embeddings_dataset[batch_size * i:batch_size * (i + 1), :] = embeddings.cpu().numpy()

    embeddings_file.close()


if __name__ == "__main__":
    configure_logging()

    parser = argparse.ArgumentParser(description="Extract features/embeddings for a given model and dataset combination")

    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--num-workers", type=int, default=18)
    parser.add_argument("--output-path", required=True, type=str, help="Path to the output file")
    parser.add_argument("--cache-dir", type=str, required=True, help="The cache directory for datasets")

    parser.add_argument("--dataset", type=DatasetType, choices=list(DatasetType), default=DatasetType.ESC_50, help="The dataset to extract features from")
    parser.add_argument("--dataset-split", type=DatasetSplit, choices=list(DatasetSplit), required=True, help="The dataset split")

    parser.add_argument("--embeddings-dim", type=int, default=768, help="The model output dimensionality")
    parser.add_argument("--model", type=ModelType, choices=list(ModelType), default=ModelType.PATCH_SSAST, help="The model used to extract features")

    main(parser.parse_args())
