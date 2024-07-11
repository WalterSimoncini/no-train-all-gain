import os
import pytest

from src.enums import DatasetSplit
from src.datasets.cub200 import CUB200DatasetFactory


def test_cub200():
    """Makes sure the CUB 200 (2011) dataset is loaded correctly"""
    base_arguments = {
        "transform": None,
        "cache_dir": os.environ["DATASET_CACHE_DIR"]
    }

    factory = CUB200DatasetFactory(cache_dir=base_arguments["cache_dir"])

    # The training and testing splits should have 5994
    # and 5794 samples respectively
    assert len(factory.load(split=DatasetSplit.TRAIN)) == 5994
    assert len(factory.load(split=DatasetSplit.TEST)) == 5794

    with pytest.raises(AssertionError):
        factory.load(split=DatasetSplit.VALID)
