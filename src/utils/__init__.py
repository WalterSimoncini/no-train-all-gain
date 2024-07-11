import torch
import torch.nn as nn

from typing import Tuple


def get_device() -> torch.device:
    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
