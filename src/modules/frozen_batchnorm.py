import torch
import torch.nn as nn

from torch.nn.functional import batch_norm


class FrozenBatchNorm2d(nn.Module):
    """
        This module performs the same operation as BatchNorm2d,
        but is always frozen, i.e. its mean and variance are
        never updated, regardless of whether the model is being
        trained or evaluated.
    """
    def __init__(
        self,
        eps: float = 1e-5,
        running_mean: torch.Tensor = None,
        running_var: torch.Tensor = None,
        weight: torch.Tensor = None,
        bias: torch.Tensor = None,
        device: torch.device = None
    ) -> None:
        super().__init__()

        self.eps = eps
        self.running_mean = running_mean.to(device)
        self.running_var = running_var.to(device)
        self.weight = weight.to(device)
        self.bias = bias.to(device)

    def forward(self, x):
        # Use running_mean and running_var as fixed values
        return batch_norm(
            x,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            training=False, 
            momentum=0,
            eps=self.eps
        )
