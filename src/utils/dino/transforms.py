"""Mostly taken from https://github.com/facebookresearch/dino"""
import torch
import random
import torch.nn as nn
import torchvision.transforms.v2 as tf

from PIL import Image
from typing import Tuple
from PIL import ImageFilter, ImageOps


class DINODataAgumentation(nn.Module):
    """
        The DINO data augmentation. By default, only random crops
        are used an style augmentations are disabled.
    """
    def __init__(
        self,
        global_crops_scale: Tuple[int, int] = (0.25, 1.0),
        local_crops_scale: Tuple[int, int] = (0.05, 0.25),
        local_crops_number: int = 10,
        crops_size: int = 224
    ):
        super().__init__()

        self.local_crops_number = local_crops_number

        flip_and_color_jitter = tf.Compose([
            tf.RandomHorizontalFlip(p=0.5),
            tf.RandomApply(
                [tf.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            tf.RandomGrayscale(p=0.2),
        ])

        normalize = tf.Compose([
            tf.ToTensor(),
            tf.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])

        self.first_global_crop = tf.Compose([
            tf.RandomResizedCrop(crops_size, scale=global_crops_scale, interpolation=Image.BICUBIC),
            # flip_and_color_jitter,
            # GaussianBlur(1.0),
            normalize
        ])

        self.second_global_crop = tf.Compose([
            tf.RandomResizedCrop(crops_size, scale=global_crops_scale, interpolation=Image.BICUBIC),
            # flip_and_color_jitter,
            # GaussianBlur(0.1),
            # Solarization(0.2),
            normalize
        ])

        self.local_crop = tf.Compose([
            # tf.RandomResizedCrop(96, scale=local_crops_scale, interpolation=Image.BICUBIC),
            tf.RandomResizedCrop(crops_size, scale=local_crops_scale, interpolation=Image.BICUBIC),
            # flip_and_color_jitter,
            # GaussianBlur(p=0.5),
            normalize,
            # DINO does not resize the local crops, but we do for compatibility with supervised ViTs
            # tf.Resize((224, 224), interpolation=Image.BICUBIC, antialias=True)
        ])

    def __call__(self, image: Image):
        crops = [self.first_global_crop(image), self.second_global_crop(image)]
        crops += [self.local_crop(image) for _ in range(self.local_crops_number)]

        # Crops is now a 12-items array (assuming the regular amount
        # of local crops is ued). We return a [12, 3, 224, 224] tensor
        return torch.cat([x.unsqueeze(dim=0) for x in crops], dim=0)


class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob

        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )


class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img
