import torch.nn as nn

from abc import ABC, abstractmethod


class ModelFactory(ABC):
    @abstractmethod
    def build(self, cache_dir: str, **kwargs) -> nn.Module:
        """
            Returns a randomly-initialized or pretrained
            model or feature extractor
        """
        raise NotImplementedError
