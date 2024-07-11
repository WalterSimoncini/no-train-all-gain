from typing import Tuple
from datasets import load_dataset
from torch.utils.data import Dataset

from src.enums import DatasetSplit


class TREC(Dataset):
    def __init__(self, split: DatasetSplit) -> None:
        assert split != DatasetSplit.VALID, "trec has no validation split!"

        split = {
            DatasetSplit.TEST: "test",
            DatasetSplit.TRAIN: "train"
        }[split]

        # https://huggingface.co/datasets/trec
        self.dataset = load_dataset("trec")[split]

    def __len__(self) -> int:
        return self.dataset.num_rows

    def __getitem__(self, idx: int) -> Tuple[str, int]:
        sample = self.dataset[int(idx)]

        # The label is binary, i.e. [0, 1]
        return sample["text"], sample["coarse_label"]
