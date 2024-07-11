import torch
import logging
import torch.nn as nn

from typing import List

from src.utils.misc import rsetattr
from src.utils.grads import get_layer
from src.modules.frozen_batchnorm import FrozenBatchNorm2d


def freeze_batchnorm_modules(model: nn.Module, device: torch.device):
    """
        Replaces all BatchNorm2d modules of the given model
        with FrozenBatchNorm2d modules, which locks the batch
        norm statistics regardless of the train/eval status
        of the model
    """
    num_frozen = 0

    for name, module in list(model.named_modules()):
        if isinstance(module, nn.BatchNorm2d):
            num_frozen += 1

            rsetattr(
                model,
                name,
                FrozenBatchNorm2d(
                    eps=module.eps,
                    running_mean=module.running_mean,
                    running_var=module.running_var,
                    weight=module.weight,
                    bias=module.bias,
                    device=device
                )
            )

    logging.info(f"frozen {num_frozen} BatchNorm2d modules")


def freeze_model(model: nn.Module, exclusions: List[str] = []) -> None:
    """
        Freezes a model, excluding the weight and bias of the
        layer paths in the exclusions array
    """
    for param in model.parameters():
        param.requires_grad_(False)

    for layer_path in exclusions:
        layer = get_layer(model=model, path=layer_path)
        layer.weight.requires_grad = True

        if hasattr(layer, "bias") and layer.bias is not None:
            layer.bias.requires_grad = True


def model_feature_dim(model: nn.Module, device: torch.device, image_size: int = 224) -> int:
    """Given a feature extractor model returns the dimensionality of its features"""
    # Forward a random image through the model to retrieve a feature
    feature = model(torch.randn(1, 3, image_size, image_size).to(device))

    # Return the feature dimensionality
    return feature.squeeze().shape[0]
