"""
    Utilities for working with SimCLR-style models.
    Some code is based on this tutorial:
    
    https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial17/SimCLR.html
"""
import torch
import torch.nn.functional as F

from torch.nn.functional import cosine_similarity


# We pick 10e-3 as it's negative enough and fits
# within the float16 range
SELF_SIMILARITY_CONSTANT = -10e3


def masked_cosine_similarity(features: torch.Tensor, n_positive_views: int) -> torch.Tensor:
    # We are only interested in the cosine similarity of the positive views
    # against all other views, so we only compute the bottom rectangle of
    # the cosine similarity matrix, selecting only rows that belong to
    # positive views
    cosine_sim = cosine_similarity(
        features[-n_positive_views:, None, :],
        features[None, :, :],
        dim=-1
    )

    # We will always have more columns than rows here
    # as the columns include the negatives, while the
    # rows only include the positive views
    n_rows, n_cols = cosine_sim.shape

    # Fill the "rightmost" diagonal with the self-similarity
    # constant, as this diagonal represents the matches of
    # each positive view with itself
    cosine_sim = torch.diagonal_scatter(
        cosine_sim,
        torch.ones(n_rows) * SELF_SIMILARITY_CONSTANT,
        n_cols - n_rows
    )

    return cosine_sim


def info_nce_loss(
    comparison_batch_features: torch.Tensor,
    positive_features: torch.Tensor,
    temperature: float = 0.07
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
    # Concatenate the batch samples and the test views in a single tensor
    features = torch.cat([
        comparison_batch_features,
        positive_features
    ], dim=0).to(torch.float16)

    features = F.normalize(features, dim=-1, p=2)

    n_positive_views = positive_features.shape[0]
    cosine_sim = masked_cosine_similarity(
        features=features,
        n_positive_views=n_positive_views
    )

    # Select only the bottom-right corner of the cosine similarity matrix, i.e.
    # the similarities between the positive views
    positive_cosine_sim = cosine_sim[-n_positive_views:, -n_positive_views:] / temperature

    # Zero out the diagonal to remove the effect of self-similarities and calculate the
    # mean self-similarities by averaging over columns. We divide by positive - 1 as
    # one element will always be zero
    nll = -(positive_cosine_sim - torch.diag(torch.diag(positive_cosine_sim))).sum(dim=1) / (n_positive_views - 1)
    nll += torch.logsumexp(cosine_sim[-n_positive_views:, :], dim=-1)

    return nll.mean()
