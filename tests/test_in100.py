import os
import torch
import pytest

from torchvision import transforms as tf

from src.enums import DatasetSplit
from src.datasets.in100 import ImageNet100


@pytest.mark.skip(reason="this requires ImageNet-100 to be downloaded and processed")
def test_imagenet_dataset():
    base_arguments = {
        "transform": None,
        "cache_dir": os.environ["DATASET_CACHE_DIR"]
    }

    # Make sure splits are loaded correctly
    train_dataset = ImageNet100(split=DatasetSplit.TRAIN, **base_arguments)
    test_dataset = ImageNet100(split=DatasetSplit.TEST, **base_arguments)

    with pytest.raises(AssertionError):
        # As this dataset has no test split we should raise an exception
        # if the test split is requested
        ImageNet100(split=DatasetSplit.VALID, **base_arguments)

    # 1300 samples for 100 classes
    assert len(train_dataset) == 100 * 1300
    # 50 samples for 100 classes
    assert len(test_dataset) == 100 * 50

    # Make sure a random sample is composed of an RGB
    # image and an integer target. We pick sample 1171
    # as this was a grayscale image originally
    image, label = train_dataset[1171]

    assert image.mode == "RGB"
    assert type(label) == int

    # Make sure transforms are applied if specified
    train_dataset = ImageNet100(
        split=DatasetSplit.TRAIN,
        **base_arguments | { "transform": tf.ToTensor() }
    )

    image, label = train_dataset[30]

    assert type(image) == torch.Tensor
