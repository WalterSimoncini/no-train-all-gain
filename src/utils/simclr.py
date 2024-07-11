"""
    Utilities for working with SimCLR-style models.
    Some code is based on this tutorial:
    
    https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial17/SimCLR.html
"""
import math
import torch
import numpy as np
import albumentations
import torch.nn as nn
import torch.nn.functional as F

from enum import Enum
from typing import List
from torch.utils.data import Dataset
from src.utils.transforms import Patchify
from albumentations.pytorch import ToTensorV2
from torch.nn.functional import cosine_similarity


# We pick 10e-3 as it's negative enough and fits
# within the float16 range
SELF_SIMILARITY_CONSTANT = -10e3


class SimCLRAugmentationType(Enum):
    DEFAULT = "default"
    RANDOM_CROPS = "random-crops"
    PATCHIFY = "patchify"


def sample_batch(dataset: Dataset, batch_size: int = 256) -> List[torch.Tensor]:
    """
        Samples a batch of size batch_size from the given
        dataset. This function is useful to obtain a fixed
        batch of negative examples for comparing a query
        sample (which may be OOD).
    """
    # Select batch_size random samples
    permuted_indices = torch.randperm(len(dataset))

    # Retrieve the transformed selected samples. This
    # assumes that the dataset transformation is wrapped
    # by a ContrastiveTransformation
    batch_samples = []
    batch_indices = permuted_indices[:batch_size]

    for i in batch_indices:
        views = dataset[i][0]
        views = torch.cat([x.unsqueeze(0) for x in views])

        batch_samples.append(views)

    return torch.cat(batch_samples, dim=0)


def precompute_comparison_batch(
    batch: torch.Tensor,
    model: nn.Module,
    device: torch.device,
    minibatch_size: int = 64
) -> torch.Tensor:
    extracted_features = []
    num_batches = math.ceil(batch.shape[0] / minibatch_size)

    # Rather than computing the whole negative batch at once
    # compute its mini-batches to avoid OOM errors
    with torch.no_grad():
        for i in range(num_batches):
            start_index = i * minibatch_size
            end_index = (i + 1) * minibatch_size

            minibatch = batch[start_index:end_index].to(device)

            extracted_features.append(model(minibatch).detach())

    return torch.cat(extracted_features, dim=0)


def masked_cosine_similarity(features: torch.Tensor, device: torch.device, n_positive_views: int) -> torch.Tensor:
    # We are only interested in the cosine similarity of the positive views
    # against all other views, so we only compute the bottom rectangle of
    # the cosine similarity matrix, selecting only rows that belong to
    # positive views
    num_features = features.shape[0]
    cosine_sim = cosine_similarity(
        features[-n_positive_views:, None, :],
        features[None, :, :],
        dim=-1
    )

    # Make the cosine similarity matrix square, as if we computed it fully.
    # This trick is done in order for masked fill to work nicely with the
    # diagonal masking
    cosine_sim = torch.cat([
        torch.zeros(num_features - n_positive_views, num_features, device=device),
        cosine_sim
    ], dim=0)

    # Mask out cosine similarity from each sample to
    # itself (by making it very negative)
    self_sim_mask = torch.eye(
        cosine_sim.shape[0],
        dtype=torch.bool,
        device=device
    )

    cosine_sim.masked_fill_(self_sim_mask, SELF_SIMILARITY_CONSTANT)

    # Finally return the bottom [positive, positive + negative]
    # rectangle we are interested in
    return cosine_sim[-n_positive_views:, :]


def info_nce_loss(
    comparison_batch_features: torch.Tensor,
    positive_features: torch.Tensor,
    device: torch.device,
    temperature: float = 0.07,
    mixing: bool = False,
    mixing_k: int = 5
) -> torch.Tensor:
    """
        Calculates the InfoNCE loss for a pair of positive
        examples (test_views) against a batch of negative
        examples (batch).

        :param batch:
            The negative examples for the loss calculation
        :param test_views:
            The positive examples for the loss calculation
    """
    # Concatenate the batch samples and the
    # test views in a single tensor
    features = torch.cat([
        comparison_batch_features,
        positive_features
    ], dim=0).to(torch.float16)

    features = F.normalize(features, dim=-1, p=2)

    n_positive_views = positive_features.shape[0]
    cosine_sim = masked_cosine_similarity(
        features=features,
        device=device,
        n_positive_views=n_positive_views
    )

    # For the hard negatives mixing we want to:
    #
    # 1. Rank hard negatives by hardest to least hard (i.e. descending similarity)
    # 2. Pick k hard negatives to mix
    # 3. Sample k beta params between (0, 0.5), as the query should have a
    #    smaller impact than the negative sample
    # 4. Generate the new mixed hard negatives!
    #
    # This feature is based on:
    #
    # Kalantidis, Y., Sariyildiz, M. B., Pion, N., Weinzaepfel, P., & Larlus, D. (2020).
    # Hard negative mixing for contrastive learning.
    # Advances in Neural Information Processing Systems, 33, 21798-21809.
    #
    # Mochi(N, s, s') where N is the number of very hard negatives (k for us)
    # s is the number of hard negatives generated by mixing other hard negatives
    # and s' is the synthetic negatives obtained from mixing the query. This
    # code block only performs the query mixing.
    if mixing:
        # Calculate the similarities between queries (positive) and
        # keys (negative) and mask out the similarities between the
        # queries and between a query and itself
        query_similarities = cosine_sim.detach()
        query_similarities[:, -n_positive_views:] = SELF_SIMILARITY_CONSTANT

        # Sort the negatives by similarity (in descending order) and
        # select the top-k hardest negatives per query
        negatives_indices = query_similarities.argsort(dim=-1, descending=True)
        negatives_indices = negatives_indices[:, :mixing_k]

        # Reshape the negative indices into an array, pick the hard negatives
        # from the features bank, reshape it to match (queries, k, feature_dim)
        # and mix the hard negatives from there
        negatives = features[negatives_indices.reshape(-1), :]
        negatives = negatives.reshape(
            negatives_indices.shape[0],
            negatives_indices.shape[1],
            -1
        )

        # Retrieve the queries and build a matrix of the same size
        # as the negatives
        queries = features[-n_positive_views:, :]
        queries = queries.unsqueeze(dim=1).repeat(1, mixing_k, 1)

        # The betas should be in the range (0, 0.5)
        betas = torch.rand(negatives.shape, device=device) / 2

        # Mix the query with the hard negatives and normalize
        query_negatives = queries * betas + (1 - betas) * negatives
        query_negatives = query_negatives.reshape(-1, query_negatives.shape[-1])

        query_negatives = F.normalize(query_negatives, dim=-1, p=2)

        # Add these new negatives to the feature matrix. We make sure that
        # the queries are at the bottom of the matrix for an easier integration
        # with the rest of the code
        original_queries = features[-n_positive_views:, :]
        original_negatives = features[:-n_positive_views, :]

        features = torch.cat([
            original_negatives,
            query_negatives,
            original_queries
        ], dim=0)

        # Recalculate the cosine similarities for the final negatives matrix
        cosine_sim = masked_cosine_similarity(
            features=features,
            device=device,
            n_positive_views=n_positive_views
        )

    # Finally calculate the InfoNCE loss
    cosine_sim = cosine_sim / temperature

    # Select only the bottom-right corner of the cosine similarity matrix, i.e.
    # the similarities between the positive views
    positive_cosine_sim = cosine_sim[-n_positive_views:, -n_positive_views:]

    # Zero out the diagonal to remove the effect of self-similarities and calculate the
    # mean self-similarities by averaging over columns. We divide by positive - 1 as
    # one element will always be zero
    nll = -(positive_cosine_sim - torch.diag(torch.diag(positive_cosine_sim))).sum(dim=1) / (n_positive_views - 1)
    nll += torch.logsumexp(cosine_sim[-n_positive_views:, :], dim=-1)

    return nll.mean()


def get_simclr_transform(
    n_views: int = 2,
    augmentation_type: SimCLRAugmentationType = SimCLRAugmentationType.DEFAULT,
    view_size: int = 224,
    **kwargs
) -> nn.Module:
    if augmentation_type == SimCLRAugmentationType.DEFAULT:
        # The data augmentation used in this implementation (with minor variations)
        # https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial17/SimCLR.html
        return ContrastiveTransformation(
            transform=albumentations.Compose([
                albumentations.HorizontalFlip(),
                albumentations.RandomResizedCrop(height=96, width=96),
                albumentations.ColorJitter(
                    brightness=0.5,
                    contrast=0.5,
                    saturation=0.5,
                    hue=0.1,
                    p=0.8
                ),
                albumentations.ToGray(p=0.2),
                albumentations.GaussianBlur(blur_limit=(9, 9), sigma_limit=(0, 0), always_apply=True),
                albumentations.Normalize(),
                albumentations.Resize(view_size, view_size),
                ToTensorV2()
            ]),
            n_views=n_views,
            **kwargs
        )
    elif augmentation_type == SimCLRAugmentationType.RANDOM_CROPS:
        # Only random-resized crops
        return ContrastiveTransformation(
            transform=albumentations.Compose([
                albumentations.RandomResizedCrop(height=view_size, width=view_size),
                albumentations.Normalize(),
                ToTensorV2()
            ]),
            n_views=n_views,
            **kwargs
        )
    elif augmentation_type == SimCLRAugmentationType.PATCHIFY:
        # Split the image into non-overlapping patches
        return Patchify(num_patches=n_views, image_size=view_size, **kwargs)
    else:
        raise ValueError(f"invalid augmentation type {augmentation_type}")


class ContrastiveTransformation(nn.Module):
    """
        Simple wrapper transformation that generates
        n_views outputs for a given sample
    """
    def __init__(self, transform: nn.Module, n_views: int = 2, **kwargs):
        super().__init__()

        self.n_views = n_views
        self.transform = transform

    def __call__(self, x):
        crops = [self.transform(image=np.array(x))["image"] for _ in range(self.n_views)]

        return torch.cat([x.unsqueeze(dim=0) for x in crops], dim=0)
