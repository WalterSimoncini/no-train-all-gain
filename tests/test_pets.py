import os
import pytest

from src.enums import DatasetSplit
from src.datasets.pets import OxfordIITPetsDatasetFactory


@pytest.mark.skip(reason="this requires the pets dataset to be downloaded and processed")
def test_oxford_iit_pets():
    """Makes sure the Oxford-IIT Pets dataset is loaded correctly"""
    base_arguments = {
        "transform": None,
        "cache_dir": os.environ["DATASET_CACHE_DIR"]
    }

    factory = OxfordIITPetsDatasetFactory(cache_dir=base_arguments["cache_dir"])

    # The training and testing splits should have 3680
    # and 3669 samples respectively
    assert len(factory.load(split=DatasetSplit.TRAIN)) == 3680
    assert len(factory.load(split=DatasetSplit.TEST)) == 3669
