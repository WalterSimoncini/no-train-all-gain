import torch.nn as nn

from abc import ABC, abstractmethod
from torchvision import transforms as tf


class ModelFactory(ABC):
    @abstractmethod
    def build(self, **kwargs) -> nn.Module:
        """
            Returns a randomly-initialized or pretrained
            model or feature extractor
        """
        raise NotImplementedError

    @abstractmethod
    def get_transform(self, **kwargs) -> nn.Module:
        raise NotImplementedError
