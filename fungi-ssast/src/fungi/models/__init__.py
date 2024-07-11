import os
import torch.nn as nn

from src.fungi.enums import ModelType

from .patch_model import SSASTPatchFactory


def get_model(type_: ModelType, cache_dir: str, **kwargs) -> nn.Module:
    """Returns an initialized model of the given kind"""
    factory = {
        ModelType.PATCH_SSAST: SSASTPatchFactory
    }[type_]()

    # Make sure that the cache directory exists, and if that
    # is not the case create it
    cache_dir = os.path.join(cache_dir, "models")
    os.makedirs(cache_dir, exist_ok=True)

    return factory.build(cache_dir=cache_dir, **kwargs)
