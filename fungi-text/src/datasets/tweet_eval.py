from typing import Tuple
from datasets import load_dataset
from torch.utils.data import Dataset

from src.enums import DatasetSplit


class TweetEval(Dataset):
    def __init__(self, split: DatasetSplit) -> None:
        split = {
            DatasetSplit.TEST: "test",
            DatasetSplit.TRAIN: "train",
            DatasetSplit.VALID: "validation"
        }[split]

        # https://huggingface.co/datasets/tweet_eval
        self.dataset = load_dataset("tweet_eval", "emoji")[split]

    def __len__(self) -> int:
        return self.dataset.num_rows

    def __getitem__(self, idx: int) -> Tuple[str, int]:
        sample = self.dataset[int(idx)]

        # The label is one of 77 classes
        return sample["text"], sample["label"]
