import torch.nn as nn

from typing import List


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


def get_layer(model: nn.Module, path: str):
    """
        Returns the layer specified by path for the
        given model. Path should be specified as a property
        path, e.g. "model.mlp.proj". Indices within Sequential
        blocks should be specified as "model.sequential.0"
    """
    path_components = path.split(".")
    layer = getattr(model, path_components[0])

    for comp in path_components[1:]:
        layer = getattr(layer, comp)

    return layer


def get_layer_grads_dim(model: nn.Module, path: str) -> tuple:
    """
        Returns the gradients dimensionality of the
        layer specified by path.
    """
    layer = get_layer(model=model, path=path)
    gradients_shape = list(layer.weight.shape)

    if hasattr(layer, "bias") and layer.bias is not None:
        # If the layer has a bias vector we add one dimension to the
        # input feature dimensionality to store the bias gradient
        gradients_shape[1] += 1

    return tuple(gradients_shape)
