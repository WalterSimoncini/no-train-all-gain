import os
import torch.nn as nn

from src.enums import DatasetSplit
from torchvision.datasets import Flowers102

from .base_factory import DatasetFactory


class Flower102DatasetFactory(DatasetFactory):
    def load(self, split: DatasetSplit, transform: nn.Module = None, **kwargs) -> nn.Module:
        split = {
            DatasetSplit.TEST: "test",
            DatasetSplit.VALID: "val",
            DatasetSplit.TRAIN: "train"
        }[split]

        return Flowers102(
            root=os.path.join(self.cache_dir, "flowers102"),
            split=split,
            download=True,
            transform=transform
        )
