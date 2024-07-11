import torch

from tests.fixtures import mock_image
from src.utils.transforms import Patchify


def test_patchify_transform(mock_image):
    transform = Patchify(num_patches=4, image_size=224, stride_scale=1)
    patches = transform(mock_image)

    assert patches.shape == torch.Size([4, 3, 224, 224])
