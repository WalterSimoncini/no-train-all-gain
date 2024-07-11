import os
import torch.nn as nn

from src.enums import DatasetSplit
from torchvision.datasets import Food101

from .base_factory import DatasetFactory


class Food101DatasetFactory(DatasetFactory):
    def load(self, split: DatasetSplit, transform: nn.Module = None, **kwargs) -> nn.Module:
        assert split != DatasetSplit.VALID, "this dataset has no validation split"

        return Food101(
            root=os.path.join(self.cache_dir, "food"),
            split=split.value,
            download=True,
            transform=transform
        )
