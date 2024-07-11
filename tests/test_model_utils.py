import torch
import torch.nn as nn

from torchvision.models import resnet50, vit_b_16
from src.utils.models import model_feature_dim, freeze_model


def test_model_feature_dim():
    """Verify that the features of a ResNet50 are found to be 2048-dimensional"""
    model = resnet50()
    model.fc = nn.Identity()

    device = torch.device("cpu")

    assert model_feature_dim(
        model=model,
        device=device
    ) == 2048


def test_freeze_model():
    """Make sure a model is frozen correctly and exclusions are respected"""
    model = vit_b_16()

    # Freeze the whole model except the last linear layer in the last transformer block
    freeze_model(model=model, exclusions=["encoder.layers.encoder_layer_11.mlp.3"])

    # Make sure the weight and bias gradients are enabled for the last layer
    assert model.encoder.layers.encoder_layer_11.mlp[3].bias.requires_grad
    assert model.encoder.layers.encoder_layer_11.mlp[3].weight.requires_grad

    # Make sure the first linear layer in the same block is frozen instead
    assert not model.encoder.layers.encoder_layer_11.mlp[0].bias.requires_grad
    assert not model.encoder.layers.encoder_layer_11.mlp[0].weight.requires_grad
