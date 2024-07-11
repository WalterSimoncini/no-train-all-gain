from enum import Enum


class DatasetSplit(Enum):
    TRAIN = "train"
    VALID = "valid"
    TEST = "test"


class DatasetType(Enum):
    TREC = "trec"
    AG_NEWS = "ag-news"
    BANKING_77 = "banking-77"
    TWEET_EVAL = "tweet-eval"
    FINE_GRAINED_SST = "fine-grained-sst"


class ModelType(Enum):
    T5_SMALL = "t5-small"
    BERT_BASE_UNCASED = "bert-base-uncased"
