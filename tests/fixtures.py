import torch
import pytest
import torch.nn as nn

from PIL import Image
from typing import List
from src.utils import get_device
from src.utils.seed import seed_everything
from torchvision.models import ViT_B_16_Weights


@pytest.fixture(autouse=True)
def setup_tests():
    """This function sets the seed before running every test"""
    seed_everything(42)
    yield


@pytest.fixture
def device() -> torch.device:
    return get_device()


@pytest.fixture
def scaler() -> torch.cuda.amp.GradScaler:
    return torch.cuda.amp.GradScaler(enabled=True)


@pytest.fixture
def dino(device) -> nn.Module:
    return torch.hub.load("facebookresearch/dino:main", "dino_vitb16").to(device)


@pytest.fixture
def sample_images() -> List[Image.Image]:
    return [
        Image.open("tests/data/mock_image.png"),
        Image.open("tests/data/shuttle.jpg")
    ]


@pytest.fixture
def images_batch(device, sample_images) -> torch.Tensor:
    transform = ViT_B_16_Weights.IMAGENET1K_V1.transforms()

    return torch.cat(
        [transform(x).unsqueeze(dim=0) for x in sample_images],
        dim=0
    ).to(device)


@pytest.fixture
def simclr_comparison_batch(device) -> torch.Tensor:
    return torch.load(
        "tests/data/comparison-batch.pth",
        map_location=device
    )


@pytest.fixture
def dino_gradients_layer_path() -> str:
    return "blocks.11.attn.proj"


@pytest.fixture
def mock_image():
    return Image.open("tests/data/mock_image.png")
