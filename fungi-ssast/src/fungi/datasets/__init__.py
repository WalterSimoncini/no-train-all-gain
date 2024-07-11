from src.dataloader import AudioDataset
from src.fungi.enums import DatasetType, DatasetSplit

from .esc50 import ESC50DatasetFactory
from .speechcommands import SpeechCommandsDatasetFactory


def load_dataset(type_: DatasetType, split: DatasetSplit, cache_dir: str, transform=None, **kwargs) -> AudioDataset:
    factory = {
        DatasetType.ESC_50: ESC50DatasetFactory,
        DatasetType.SPEECH_COMMANDS: SpeechCommandsDatasetFactory
    }[type_](cache_dir=cache_dir)

    return factory.load(
        dataset_type=type_,
        split=split,
        transform=transform,
        **kwargs
    )
