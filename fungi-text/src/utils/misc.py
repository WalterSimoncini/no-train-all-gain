import os
import torch
import random
import logging
import functools
import numpy as np

from typing import List, Dict


def seed_everything(seed: int):
    """
        Set seeds for python's random function, the numpy seed,
        torch and configure CUDA to use the deterministic backend
    """
    random.seed(seed)
    np.random.seed(seed)

    os.environ["PYTHONHASHSEED"] = str(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def configure_logging():
    logging.basicConfig(
        format="[%(asctime)s:%(levelname)s]: %(message)s",
        level=logging.INFO
    )


def rsetattr(obj, attr, val):
    """
        Sets a nested attribute, e.g. model.encoder. Code based on:
        
        https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects/31174427?noredirect=1#comment86638618_31174427
    """
    pre, _, post = attr.rpartition('.')

    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    """
        Gets a nested attribute, e.g. model.encoder. Code based on:
        
        https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects/31174427?noredirect=1#comment86638618_31174427
    """
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split('.'))


def parse_gradient_targets(targets: List[str]) -> Dict[str, str]:
    """
        Parses the targets for gradient extraction, which are
        specified in the command line argument as an array of
        items in the format layer_path:out_dataset and returns a
        mapping from the layer_path to the dataset
    """
    return dict([
        # Each layer is specified as layer_path:dataset_name,
        # so to get the keys and values we can simply split
        # the string by the ":" character
        target.split(":") for target in targets
    ])
