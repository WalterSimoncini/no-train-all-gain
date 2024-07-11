from torch.utils.data import Dataset

from .trec import TREC
from .ag_news import AGNews
from .banking import Banking77
from .tweet_eval import TweetEval
from .fine_sst import FineGrainedSST
from src.enums import DatasetSplit, DatasetType


def load_dataset(type_: DatasetType, split: DatasetSplit) -> Dataset:
    """Returns a torch Dataset of the given type and split"""
    return {
        DatasetType.FINE_GRAINED_SST: FineGrainedSST,
        DatasetType.AG_NEWS: AGNews,
        DatasetType.TREC: TREC,
        DatasetType.BANKING_77: Banking77,
        DatasetType.TWEET_EVAL: TweetEval
    }[type_](split=split)
