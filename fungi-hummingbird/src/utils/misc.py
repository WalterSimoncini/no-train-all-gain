import os
import torch
import random
import logging
import numpy as np
import torch.nn as nn


def seed_everything(seed: int):
    """
        Set seeds for python's random function, the numpy seed,
        torch and configure CUDA to use the deterministic backend
    """
    random.seed(seed)
    np.random.seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def model_feature_dim(model: nn.Module, device: torch.device, image_size: int = 224) -> int:
    """Given a feature extractor model returns the dimensionality of its features"""
    # Forward a random image through the model to retrieve a feature
    feature = model(torch.randn(1, 3, image_size, image_size).to(device))

    # Return the feature dimensionality
    return feature.squeeze().shape[0]


def configure_logging():
    logging.basicConfig(
        format="[%(asctime)s:%(levelname)s]: %(message)s",
        level=logging.INFO
    )
