import os
import torch.nn as nn

from typing import Tuple
from PIL.Image import Image
from datasets import load_from_disk
from torch.utils.data import Dataset

from src.enums import DatasetSplit
from .base_factory import DatasetFactory


class CUB200DatasetFactory(DatasetFactory):
    def load(
        self,
        split: DatasetSplit,
        transform: nn.Module = None,
        **kwargs
    ) -> nn.Module:
        return CUB200(
            split=split,
            transform=transform,
            cache_dir=self.cache_dir
        )


class CUB200(Dataset):
    """
        The CUB 200 (2011) dataset. Follow the instructions in
        the README.md file to setup this dataset
    """
    def __init__(
        self,
        split: DatasetSplit,
        transform: nn.Module,
        cache_dir: str = None
    ) -> None:
        """
            Initializes a CUB 200 (2011) dataset. The dataset is assumed to
            be in the cache_dir/cub-hf folder, which should contain a HuggingFace
            dataset generated using the tools/cub_2011.py script.
        """
        assert split != DatasetSplit.VALID, "CUB 200 has no validation split"

        self.transform = transform
        self.dataset = load_from_disk(
            os.path.join(cache_dir, "cub-hf")
        )[split.value]

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
