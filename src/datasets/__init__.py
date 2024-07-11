import os
import torch.nn as nn

from src.enums import DatasetType, DatasetSplit

from .in1k import IN1KDatasetFactory
from .cifar10 import CIFAR10DatasetFactory
from .cifar100 import CIFAR100DatasetFactory
from .in100 import ImageNet100DatasetFactory
from .textures import TexturesDatasetFactory
from .flowers102 import Flower102DatasetFactory
from .cars import StanfordCarsDatasetFactory
from .food import Food101DatasetFactory
from .aircraft import FGVCAircraftDatasetFactory
from .pets import OxfordIITPetsDatasetFactory
from .cub200 import CUB200DatasetFactory
from .eurosat import EuroSATDatasetFactory
from .oxford_paris import ParisLandmarksFactory, OxfordLandmarksFactory


def load_dataset(
    type_: DatasetType,
    split: DatasetSplit,
    cache_dir: str,
    transform: nn.Module = None,
    **kwargs
) -> nn.Module:
    """Returns a torch Dataset of the given type and split"""
    # Create the cache directory if it does not exists already
    os.makedirs(cache_dir, exist_ok=True)

    factory = {
        DatasetType.CIFAR10: CIFAR10DatasetFactory,
        DatasetType.CIFAR100: CIFAR100DatasetFactory,
        DatasetType.IN1K: IN1KDatasetFactory,
        DatasetType.IMAGENET_100: ImageNet100DatasetFactory,
        DatasetType.TEXTURES: TexturesDatasetFactory,
        DatasetType.FLOWERS_102: Flower102DatasetFactory,
        DatasetType.STANFORD_CARS: StanfordCarsDatasetFactory,
        DatasetType.FOOD_101: Food101DatasetFactory,
        DatasetType.FGVC_AIRCRAFT: FGVCAircraftDatasetFactory,
        DatasetType.OXFORD_IIT_PETS: OxfordIITPetsDatasetFactory,
        DatasetType.CUB_200_2011: CUB200DatasetFactory,
        DatasetType.EUROSAT: EuroSATDatasetFactory,
        DatasetType.PARIS_LANDMARKS: ParisLandmarksFactory,
        DatasetType.OXFORD_LANDMARKS: OxfordLandmarksFactory
    }[type_](cache_dir=cache_dir)

    return factory.load(
        transform=transform,
        split=split,
        dataset_type=type_,
        **kwargs
    )
