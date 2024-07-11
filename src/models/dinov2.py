import torch
import torch.nn as nn

from torchvision import transforms as tf

from src.enums import ModelType
from .base_factory import ModelFactory


class DINOV2Factory(ModelFactory):
    def build(self, model_type: ModelType, **kwargs) -> nn.Module:
        return torch.hub.load("facebookresearch/dinov2", model_type.value)

    def get_transform(self, **kwargs) -> nn.Module:
        # https://github.com/facebookresearch/dinov2/blob/e1277af2ba9496fbadf7aec6eba56e8d882d1e35/dinov2/data/transforms.py#L77
        return tf.Compose([
            tf.Resize(256, interpolation=tf.InterpolationMode.BICUBIC),
            tf.CenterCrop(224),
            tf.ToTensor(),
            tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
