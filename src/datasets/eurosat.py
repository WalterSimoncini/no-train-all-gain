import os
import ssl
import torch.nn as nn

from src.enums import DatasetSplit
from torch.utils.data import Subset
from torchvision.datasets import EuroSAT
from sklearn.model_selection import StratifiedShuffleSplit

from .base_factory import DatasetFactory


class EuroSATDatasetFactory(DatasetFactory):
    def load(self, split: DatasetSplit, transform: nn.Module = None, random_state: int = 42, **kwargs) -> nn.Module:
        assert split != DatasetSplit.VALID, "EuroSAT has no validation split!"

        root_dir = os.path.join(self.cache_dir, "eurosat")

        if not os.path.exists(root_dir):
            # Needed for the EuroSAT download to work correctly
            ssl._create_default_https_context = ssl._create_unverified_context

        dataset = EuroSAT(
            root=root_dir,
            download=True,
            transform=transform
        )

        splitter = StratifiedShuffleSplit(n_splits=1, train_size=0.8, random_state=random_state)
        training_indices, test_indices = list(splitter.split(dataset.targets, dataset.targets))[0]

        if split == DatasetSplit.TRAIN:
            return Subset(dataset=dataset, indices=training_indices)
        elif split == DatasetSplit.TEST:
            return Subset(dataset=dataset, indices=test_indices)
        else:
            raise ValueError(f"invalid split {split}")
