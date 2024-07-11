import os
import torch.nn as nn

from src.dataloader import AudioDataset
from src.fungi.enums import DatasetSplit

from .base_factory import DatasetFactory


class SpeechCommandsDatasetFactory(DatasetFactory):
    def load(self, split: DatasetSplit, transform: nn.Module, **kwargs) -> nn.Module:
        base_dir = os.path.join(self.cache_dir, "speechcommands")

        json_file = {
            DatasetSplit.TRAIN: "speechcommand_train_data.json",
            DatasetSplit.TEST: "speechcommand_eval_data.json",
            DatasetSplit.VALID: "speechcommand_valid_data.json"
        }[split]

        return AudioDataset(
            transform=transform,
            dataset_json_file=os.path.join(base_dir, "datafiles", json_file),
            label_csv=os.path.join(base_dir, "speechcommands_class_labels_indices.csv"),
            audio_conf={
                "num_mel_bins": 128,
                "target_length": 128,
                "freqm": 0,
                "timem": 0,
                "mixup": 0,
                "dataset": "speechcommands",
                "mode": "evaluation",
                "mean": -6.845978,
                "std": 5.5654526,
                "noise": False
            }
        )
