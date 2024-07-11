"""Mostly taken from https://github.com/facebookresearch/swav"""
import torch
import torchvision.transforms.v2 as tf

from PIL import Image
from typing import List
from src.utils.dino.transforms import GaussianBlur


class DeepClusterDataAgumentation():
    """
        The DeepCluster data augmentation. By default, only random crops
        are used an style augmentations are disabled. Mostly taken from
        https://github.com/facebookresearch/swav
    """
    def __init__(
        self,
        crops_size: List[int] = [224, 224],
        num_crops: List[int] = [2, 6],
        min_scale_crops: List[int] = [0.14, 0.05],
        max_scale_crops: List[int] = [1.0, 0.14]
    ):
        assert len(crops_size) == len(num_crops)
        assert len(min_scale_crops) == len(num_crops)
        assert len(max_scale_crops) == len(num_crops)

        self.transforms = []

        # color_transform = color_distortion_augmentation()

        for i in range(len(crops_size)):
            self.transforms.extend([
                tf.Compose([
                    tf.RandomResizedCrop(
                        crops_size[i],
                        scale=(min_scale_crops[i], max_scale_crops[i])
                    ),
                    # tf.RandomHorizontalFlip(p=0.5),
                    # color_transform,
                    tf.ToTensor(),
                    tf.Normalize(mean=[0.485, 0.456, 0.406], std=[0.228, 0.224, 0.225])
                ])
            ] * num_crops[i])

    def __call__(self, x: Image.Image) -> torch.Tensor:
        return torch.cat(
            [transform(x).unsqueeze(dim=0) for transform in self.transforms],
            dim=0
        )


def color_distortion_augmentation(s=1.0):
    """
        :param s: the strength of the color distortion
    """
    return tf.Compose([
        tf.RandomApply([
            tf.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        ], p=0.8),
        tf.RandomGrayscale(p=0.2),
        GaussianBlur()
    ])
