import os

from src.enums import DatasetSplit
from src.datasets.cars import StanfordCarsDatasetFactory


def test_stanford_cars():
    """Makes sure the Stanford Cars dataset is loaded correctly"""
    base_arguments = {
        "transform": None,
        "cache_dir": os.environ["DATASET_CACHE_DIR"]
    }

    factory = StanfordCarsDatasetFactory(cache_dir=base_arguments["cache_dir"])
    dataset = factory.load(split=DatasetSplit.TRAIN)

    assert len(dataset) == 8144
