from enum import Enum


class DatasetType(Enum):
    ESC_50 = "esc-50"
    SPEECH_COMMANDS = "speech-commands"


class DatasetSplit(Enum):
    TRAIN = "train"
    VALID = "valid"
    TEST = "test"


class ModelType(Enum):
    PATCH_SSAST = "patch-ssast"
