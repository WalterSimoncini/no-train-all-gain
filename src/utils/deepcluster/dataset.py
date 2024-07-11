import torch

from PIL.Image import Image
from torch.utils.data import Dataset
from typing import Tuple, List, Union

from .transforms import DeepClusterDataAgumentation


class MultiCropDataset(Dataset):
    """
        Multi-Crop dataset for DeepCluster. Partially inspired by
        https://github.com/facebookresearch/swav
    """
    def __init__(
        self,
        dataset,
        crops_size: List[int] = [224, 224],
        num_crops: List[int] = [2, 6],
        min_scale_crops: List[int] = [0.14, 0.05],
        max_scale_crops: List[int] = [1.0, 0.14],
        return_index: bool = False
    ):
        assert dataset.transform is None, "the input dataset should not use any transform"

        self.dataset = dataset
        self.return_index = return_index
        self.transform = DeepClusterDataAgumentation(
            crops_size=crops_size,
            num_crops=num_crops,
            min_scale_crops=min_scale_crops,
            max_scale_crops=max_scale_crops
        )

    def __len__(self) -> int:
        return len(self.dataset)

    def __getitem__(self, idx: int) -> Union[Image, Tuple[Image, int]]:
        """Returns a tensor of shape [num_crops, 3, 224, 224]"""
        image, _ = self.dataset[int(idx)]

        # self.transforms.transforms is an array of torchvision transforms, of
        # size sum(num_crops), i.e. by default we will have two global
        # transformations and six local ones
        crops = [transform(image).unsqueeze(dim=0) for transform in self.transform.transforms]
        crops = torch.cat(crops, dim=0)

        if self.return_index:
            return crops, idx

        return crops
