"""
    This script generates a HuggingFace dataset
    for ImageNet100
"""
import os
import json
import glob
import datasets
import argparse

from PIL import Image
from typing import List
from datasets import Dataset, Features, DatasetDict


def get_data_directories(root_dir: str, split: str) -> List[str]:
    """
        Returns all the class directories paths for a given
        split of ImageNet-100. The original data is structured
        as follows (for the train split)

        - train.X1
            - class1
                - image1.JPG
                - image2.JPG
            - class2
                - ...
        - train.X2
            - ...
    """
    training_class_dirs = []

    # The dataset might have multiple folders for a single split.
    # This line finds all the root folders for that split
    training_data_dirs = glob.glob(os.path.join(root_dir, f"{split}.X*"))

    # List all the class directories in each of the split shards
    for path in training_data_dirs:
        training_class_dirs.extend(
            glob.glob(os.path.join(path, "*"))
        )

    # Remove the root directories
    return list(set(training_class_dirs) - set(training_data_dirs))


def generate_split(data_directories: List[str], metadata: dict):
    """
        Iterate over all the shards folder, and for each folder
        iterate over the classes (i.e. sub-folders) contained in
        it and return individual images (as a generator). An image's
        label corresponds to its directory name. The metadata
        dictionary maps a directory name to its ImageNet label
    """
    for data_dir in data_directories:
        _, label_key = os.path.split(data_dir)
        directory_label = metadata[label_key]

        image_paths = glob.glob(os.path.join(data_dir, "*"))

        for image_path in image_paths:
            yield {
                "image": Image.open(image_path),
                "label": directory_label
            }


def main(args):
    metadata = json.loads(open(os.path.join(args.root_dir, "Labels.json")).read())

    val_directories = get_data_directories(root_dir=args.root_dir, split="val")
    train_directories = get_data_directories(root_dir=args.root_dir, split="train")

    features = Features({
        "image": datasets.Image(),
        "label": datasets.ClassLabel(names=list(metadata.values()))
    })

    out_dataset = DatasetDict({
        "train": Dataset.from_generator(
            num_proc=args.num_proc,
            generator=generate_split,
            features=features,
            cache_dir=args.cache_dir,
            gen_kwargs={
                "metadata": metadata,
                "data_directories": train_directories
            }
        ),
        "valid": Dataset.from_generator(
            num_proc=args.num_proc,
            generator=generate_split,
            features=features,
            cache_dir=args.cache_dir,
            gen_kwargs={
                "metadata": metadata,
                "data_directories": val_directories
            }
        )
    })

    out_dataset.save_to_disk(args.output_folder, num_proc=args.num_proc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Subsample ImageNet-1k by picking 10 pre-determined classes")

    parser.add_argument("--num-proc", type=int, default=18, help="Number of processors used for generating and saving the dataset")
    parser.add_argument("--output-folder", type=str, required=True, help="Where to save the processed dataset")
    parser.add_argument("--cache-dir", type=str, required=True, help="The cache directory for datasets")
    parser.add_argument("--root-dir", type=str, required=True, help="The root directory where the ImageNet100 dataset is located")

    main(parser.parse_args())
