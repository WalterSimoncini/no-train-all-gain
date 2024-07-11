import os
import torch.nn as nn

from typing import Tuple
from src.enums import ModelType

from .dino import DINOFactory
from .dinov2 import DINOV2Factory
from .mae import MAEViT16BFactory
from .clip_vision import CLIPViT16BFactory, EVACLIPViT16BFactory
from .vit import (
    ViT16BFactory,
    ViT32BFactory,
    ViT16SFactory,
    MoCoV3ViT16BFactory,
    ViT16BAugRegIN1KFactory,
    ViT16SAugRegIN21KFactory,
    ViT16BAugRegIN21KFactory,
    ViT16LAugRegIN21KFactory
)


def get_model(type_: ModelType, cache_dir: str, **kwargs) -> Tuple[nn.Module, nn.Module]:
    """Returns an initialized model of the given kind"""
    factory = {
        ModelType.DINO_VIT_16_B: DINOFactory,
        ModelType.DINO_V2_VIT_14_B: DINOV2Factory,
        ModelType.CLIP_VIT_16_B: CLIPViT16BFactory,
        ModelType.EVA_CLIP_VIT_16_B: EVACLIPViT16BFactory,
        ModelType.VIT_S_16: ViT16SFactory,
        ModelType.VIT_B_16: ViT16BFactory,
        ModelType.VIT_B_32: ViT32BFactory,
        ModelType.MAE_VIT_B_16: MAEViT16BFactory,
        ModelType.MOCO_V3_VIT_B_16: MoCoV3ViT16BFactory,
        ModelType.VIT_B_16_AUGREG_IN1K: ViT16BAugRegIN1KFactory,
        ModelType.VIT_S_16_AUGREG_IN21K: ViT16SAugRegIN21KFactory,
        ModelType.VIT_B_16_AUGREG_IN21K: ViT16BAugRegIN21KFactory,
        ModelType.VIT_L_16_AUGREG_IN21K: ViT16LAugRegIN21KFactory
    }[type_]()

    # Make sure that the cache directory exists, and if that
    # is not the case create it
    cache_dir = os.path.join(cache_dir, "models")
    os.makedirs(cache_dir, exist_ok=True)

    model = factory.build(model_type=type_, cache_dir=cache_dir, **kwargs)
    transform = factory.get_transform(**kwargs)

    return model, transform
