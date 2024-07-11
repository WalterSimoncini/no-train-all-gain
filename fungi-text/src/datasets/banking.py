from typing import Tuple
from datasets import load_dataset
from torch.utils.data import Dataset

from src.enums import DatasetSplit


class Banking77(Dataset):
    def __init__(self, split: DatasetSplit) -> None:
        assert split != DatasetSplit.VALID, "banking77 has no validation split!"

        split = {
            DatasetSplit.TEST: "test",
            DatasetSplit.TRAIN: "train"
        }[split]

        # https://huggingface.co/datasets/PolyAI/banking77
        self.dataset = load_dataset("PolyAI/banking77")[split]

    def __len__(self) -> int:
        return self.dataset.num_rows

    def __getitem__(self, idx: int) -> Tuple[str, int]:
        sample = self.dataset[int(idx)]

        # The label is one of 77 classes
        return sample["text"], sample["label"]
