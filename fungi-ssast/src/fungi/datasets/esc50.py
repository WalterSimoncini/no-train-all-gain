import os
import torch.nn as nn

from src.dataloader import AudioDataset
from src.fungi.enums import DatasetSplit

from .base_factory import DatasetFactory


class ESC50DatasetFactory(DatasetFactory):
    def load(self, split: DatasetSplit, transform: nn.Module, **kwargs) -> nn.Module:
        assert split != DatasetSplit.VALID, "This dataset has no validation split"

        base_dir = os.path.join(self.cache_dir, "esc-50")

        json_file = {
            DatasetSplit.TRAIN: "esc_train_data_1.json",
            DatasetSplit.TEST: "esc_eval_data_1.json"
        }[split]

        return AudioDataset(
            transform=transform,
            dataset_json_file=os.path.join(base_dir, "datafiles", json_file),
            label_csv=os.path.join(base_dir, "esc_class_labels_indices.csv"),
            audio_conf={
                "num_mel_bins": 128,
                "target_length": 512,
                "freqm": 0,
                "timem": 0,
                "mixup": 0,
                "dataset": "esc50",
                "mode": "evaluation",
                "mean": -6.6268077,
                "std": 5.358466,
                "noise": False
            }
        )
