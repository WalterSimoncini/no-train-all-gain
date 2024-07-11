import os
import torch.nn as nn

from src.enums import DatasetSplit
from torchvision.datasets import StanfordCars

from .base_factory import DatasetFactory


class StanfordCarsDatasetFactory(DatasetFactory):
    # FIXME: This dataset is currently not available from torchvsion. See the instructions
    # here: https://github.com/pytorch/vision/issues/7545 or follow the procedure in the
    # README.md file to setup this dataset correctly
    def load(self, split: DatasetSplit, transform: nn.Module = None, **kwargs) -> nn.Module:
        assert split != DatasetSplit.VALID, "this dataset has no validation split"

        return StanfordCars(
            root=os.path.join(self.cache_dir, "cars"),
            split=split.value,
            transform=transform
        )
