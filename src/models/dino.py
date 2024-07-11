import torch
import torch.nn as nn

from torchvision import transforms as tf

from src.enums import ModelType
from .base_factory import ModelFactory


class DINOFactory(ModelFactory):
    def build(self, model_type: ModelType, **kwargs) -> nn.Module:
        return torch.hub.load("facebookresearch/dino:main", model_type.value)

    def get_transform(self, **kwargs) -> nn.Module:
        # https://github.com/facebookresearch/dino/blob/main/eval_knn.py
        return tf.Compose([
            tf.Resize(256, interpolation=3),
            tf.CenterCrop(224),
            tf.ToTensor(),
            tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
