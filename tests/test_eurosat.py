import os

from src.enums import DatasetSplit
from src.datasets.eurosat import EuroSATDatasetFactory


def test_eurosat_dataset():
    """Makes sure the EuroSAT dataset is loaded correctly"""
    base_arguments = {
        "transform": None,
        "cache_dir": os.environ["DATASET_CACHE_DIR"]
    }

    factory = EuroSATDatasetFactory(cache_dir=base_arguments["cache_dir"])

    # The original dataset has only a training split, and we do an 80/20
    # split as in the original paper. The dataset has 27000 samples, so
    # the split should result in (21600, 5400) samples for training and
    # testing splits respectively
    test_split = factory.load(split=DatasetSplit.TEST)

    assert len(factory.load(split=DatasetSplit.TRAIN)) == 21600
    assert len(factory.load(split=DatasetSplit.TEST)) == 5400

    # Make sure that retrieving the test split twice results
    # in the same subset indices
    assert (test_split.indices == factory.load(split=DatasetSplit.TEST).indices).all()
