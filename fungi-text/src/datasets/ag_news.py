from typing import Tuple
from datasets import load_dataset
from torch.utils.data import Dataset

from src.enums import DatasetSplit


class AGNews(Dataset):
    def __init__(self, split: DatasetSplit) -> None:
        assert split != DatasetSplit.VALID, "ag news has no validation split!"

        split = {
            DatasetSplit.TEST: "test",
            DatasetSplit.TRAIN: "train"
        }[split]

        # https://huggingface.co/datasets/ag_news
        self.dataset = load_dataset("ag_news")[split]

    def __len__(self) -> int:
        return self.dataset.num_rows

    def __getitem__(self, idx: int) -> Tuple[str, int]:
        sample = self.dataset[int(idx)]

        # The label is a class in the interval [0, 3]
        return sample["text"], sample["label"]
