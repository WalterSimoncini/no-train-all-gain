import numpy as np

from typing import Tuple
from datasets import load_dataset
from torch.utils.data import Dataset

from src.enums import DatasetSplit


class FineGrainedSST(Dataset):
    def __init__(self, split: DatasetSplit) -> None:
        split = {
            DatasetSplit.TEST: "test",
            DatasetSplit.TRAIN: "train",
            DatasetSplit.VALID: "validation"
        }[split]

        # https://huggingface.co/datasets/sst
        self.dataset = load_dataset("sst")[split]
        self.bins = np.linspace(0, 1, 6)

    def __len__(self) -> int:
        return self.dataset.num_rows

    def __getitem__(self, idx: int) -> Tuple[str, int]:
        sample = self.dataset[int(idx)]
        sentence = sample["sentence"]

        # Bin continuous labels between [0, 4]
        label = np.digitize(sample["label"], self.bins) - 1

        return sentence, label
