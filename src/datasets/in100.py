import os
import torch.nn as nn

from typing import Tuple
from PIL.Image import Image
from datasets import load_from_disk
from torch.utils.data import Dataset

from src.enums import DatasetSplit
from .base_factory import DatasetFactory


class ImageNet100DatasetFactory(DatasetFactory):
    def load(
        self,
        split: DatasetSplit,
        transform: nn.Module = None,
        **kwargs
    ) -> nn.Module:
        return ImageNet100(
            split=split,
            transform=transform,
            cache_dir=self.cache_dir
        )


class ImageNet100(Dataset):
    def __init__(
        self,
        split: DatasetSplit,
        transform: nn.Module,
        cache_dir: str = None
    ) -> None:
        """
            Initializes a ImageNet-100 dataset. The dataset is assumed to
            be in the cache_dir/imagenet100-hf folder, which should contain
            a HuggingFace dataset generated using the tools/imagenet100.py
            script.
        """
        assert split != DatasetSplit.VALID, "ImageNet-100 has no validation split"

        # The validation set of ImageNet is generally used as the test set
        split = {
            DatasetSplit.TEST: "valid",
            DatasetSplit.TRAIN: "train"
        }[split]

        self.transform = transform
        self.dataset = load_from_disk(
            os.path.join(cache_dir, "imagenet100-hf")
        )[split]

    def __len__(self) -> int:
        return self.dataset.num_rows

    def __getitem__(self, idx: int) -> Tuple[Image, int]:
        example = self.dataset[int(idx)]
        image = example["image"]

        if image.mode != "RGB":
            # Convert grayscale images to RGB
            image = image.convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, example["label"]
