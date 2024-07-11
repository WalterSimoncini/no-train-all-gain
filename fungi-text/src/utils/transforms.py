import torch.nn as nn
import nlpaug.augmenter.word as naw

from typing import List


class ContrastiveRandomCrop(nn.Module):
    """
        Augmentation that creates multiple views of a given
        sentence by randomly removing words from it. Each
        word has a 10% probability of being removed.
    """
    def __init__(self, num_views: int = 2, aug_p: float = 0.1):
        super().__init__()

        self.num_views = num_views
        self.augmentation = naw.RandomWordAug(action="delete", aug_p=aug_p)

    def __call__(self, texts: List[str]) -> List[List[str]]:
        augmented_texts = []

        for text in texts:
            augmented_texts.append([
                self.augmentation.augment(text).pop() for _ in range(self.num_views)
            ])

        return augmented_texts
