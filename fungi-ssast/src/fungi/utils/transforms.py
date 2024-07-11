import torch
import numpy as np
import torch.nn as nn


class ContrastiveNoiseTransform(nn.Module):
    """
        Augmentation that creates multiple views of a given
        audio clip by adding light uniform noise and shifting it
    """
    def __init__(self, num_views: int = 2):
        super().__init__()

        self.num_views = num_views

    def __call__(self, clip: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [self.add_noise(clip).unsqueeze(dim=0) for _ in range(self.num_views)],
            dim=0
        )

    def add_noise(self, clip: torch.Tensor) -> torch.Tensor:
        """
            Add uniform noise to the clip and shift it by up to 10 points
            in the positive or negative direction
        """
        clip = clip + torch.rand(clip.shape[0], clip.shape[1]) * np.random.rand() / 10

        return torch.roll(clip, np.random.randint(-10, 10), 0)
