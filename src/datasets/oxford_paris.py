import os
import pickle

import torch.nn as nn

from PIL import Image, ImageFile
from torch.utils.data import Dataset

from src.enums import DatasetSplit
from .base_factory import DatasetFactory


class OxfordLandmarksFactory(DatasetFactory):
    def load(self, split: DatasetSplit, transform: nn.Module = None, imsize: int = None, **kwargs) -> nn.Module:
        assert split != DatasetSplit.VALID, "The Oxford landmarks dataset has no validation split"

        version = {
            DatasetSplit.TRAIN: "train",
            DatasetSplit.TEST: "query"
        }[split]

        return OxfordParisDataset(
            dir_main=os.path.join(self.cache_dir, "retrieval-landmarks"),
            dataset="roxford5k",
            split=version,
            transform=transform,
            imsize=imsize
        )


class ParisLandmarksFactory(DatasetFactory):
    def load(self, split: DatasetSplit, transform: nn.Module = None, imsize: int = None, **kwargs) -> nn.Module:
        assert split != DatasetSplit.VALID, "The Paris landmarks dataset has no validation split"

        version = {
            DatasetSplit.TRAIN: "train",
            DatasetSplit.TEST: "query"
        }[split]

        return OxfordParisDataset(
            dir_main=os.path.join(self.cache_dir, "retrieval-landmarks"),
            dataset="rparis6k",
            split=version,
            transform=transform,
            imsize=imsize
        )


class OxfordParisDataset(Dataset):
    """
        Oxford and Paris Landmark dataset. Mostly copied from
        https://github.com/facebookresearch/dino/blob/main/eval_image_retrieval.py
    """
    def __init__(self, dir_main, dataset, split, transform=None, imsize=None):
        if dataset not in ['roxford5k', 'rparis6k']:
            raise ValueError('Unknown dataset: {}!'.format(dataset))

        # loading imlist, qimlist, and gnd, in cfg as a dict
        gnd_fname = os.path.join(dir_main, dataset, 'gnd_{}.pkl'.format(dataset))

        with open(gnd_fname, 'rb') as f:
            cfg = pickle.load(f)

        cfg['gnd_fname'] = gnd_fname
        cfg['ext'] = '.jpg'
        cfg['qext'] = '.jpg'
        cfg['dir_data'] = os.path.join(dir_main, dataset)
        cfg['dir_images'] = os.path.join(cfg['dir_data'], 'jpg')
        cfg['n'] = len(cfg['imlist'])
        cfg['nq'] = len(cfg['qimlist'])
        cfg['dataset'] = dataset

        self.cfg = cfg

        self.samples = cfg["qimlist"] if split == "query" else cfg["imlist"]
        self.transform = transform
        self.imsize = imsize

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path = os.path.join(self.cfg["dir_images"], self.samples[index] + ".jpg")
        ImageFile.LOAD_TRUNCATED_IMAGES = True

        with open(path, 'rb') as f:
            img = Image.open(f)
            img = img.convert('RGB')
        if self.imsize is not None:
            img.thumbnail((self.imsize, self.imsize), Image.ANTIALIAS)
        if self.transform is not None:
            img = self.transform(img)

        return img, index
