"""
    This script generates a HuggingFace dataset for
    the CUB 200 (2011) dataset
"""
import os
import datasets
import argparse

from PIL import Image
from typing import List
from typing import TextIO, List
from datasets import Dataset, Features, DatasetDict


def generate_split(images_base_path: str, image_paths: List[str]):
    """
        Given the path to the folder containing the dataset images and
        the relative image paths generates rows for an HuggingFace
        dataset containing images and their class. The class name
        corresponds to the head of each relative image path
    """
    for path in image_paths:
        yield {
            "image": Image.open(os.path.join(images_base_path, path)),
            "label": os.path.split(path)[0]
        }


def readlines(in_file: TextIO) -> List[str]:
    """
        Reads all lines of the given input file-like and
        returns them, excluding blank lines
    """
    return [x for x in in_file.read().split("\n") if x.strip() != ""]


def main(args):
    images_folder = os.path.join(args.root_dir, "images")

    class_names = readlines(open(os.path.join(args.root_dir, "classes.txt")))
    class_names = [x.split(" ")[1] for x in class_names]

    # Create a list of (image id, path) tuples. The class information
    # is contained in the image path
    images_paths = readlines(open(os.path.join(args.root_dir, "images.txt")))
    images_paths = [x.split(" ") for x in images_paths]

    # Create a list of (image id, split). Split being "1" indicates that the
    # image belongs to the training set, "0" to the test set
    splits_mapping = readlines(open(os.path.join(args.root_dir, "train_test_split.txt")))
    splits_mapping = [x.split(" ") for x in splits_mapping]

    # Create sets containing the training/testing image ids
    test_image_ids = set([image_id for (image_id, split) in splits_mapping if split == "0"])
    train_image_ids = set([image_id for (image_id, split) in splits_mapping if split == "1"])

    # Create arrays of testing and training image paths
    test_image_paths = [path for (image_id, path) in images_paths if image_id in test_image_ids]
    train_image_paths = [path for (image_id, path) in images_paths if image_id in train_image_ids]

    print(f"the training dataset has {len(train_image_paths)} samples")
    print(f"the test dataset has {len(test_image_paths)} samples")

    features = Features({
        "image": datasets.Image(),
        "label": datasets.ClassLabel(names=class_names)
    })

    out_dataset = DatasetDict({
        "train": Dataset.from_generator(
            num_proc=args.num_proc,
            generator=generate_split,
            features=features,
            cache_dir=args.cache_dir,
            gen_kwargs={
                "images_base_path": images_folder,
                "image_paths": train_image_paths
            }
        ),
        "test": Dataset.from_generator(
            num_proc=args.num_proc,
            generator=generate_split,
            features=features,
            cache_dir=args.cache_dir,
            gen_kwargs={
                "images_base_path": images_folder,
                "image_paths": test_image_paths
            }
        ),
    })

    out_dataset.save_to_disk(args.output_folder, num_proc=args.num_proc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a HuggingFace dataset for the CUB 200 (2011) Pets dataset")

    parser.add_argument("--num-proc", type=int, default=18, help="Number of processors used for generating and saving the dataset")
    parser.add_argument("--output-folder", type=str, required=True, help="Where to save the processed dataset")
    parser.add_argument("--cache-dir", type=str, required=True, help="The cache directory for datasets")
    parser.add_argument("--root-dir", type=str, required=True, help="Path to the 'CUB_200_2011' directory")

    main(parser.parse_args())
