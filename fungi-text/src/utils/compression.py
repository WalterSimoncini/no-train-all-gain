"""
    Implements the random projection dimensionality reduction technique described in

    @inproceedings{achlioptas2001database,
        title={Database-friendly random projections},
        author={Achlioptas, Dimitris},
        booktitle={Proceedings of the twentieth ACM SIGMOD-SIGACT-SIGART symposium on Principles of database systems},
        pages={274--281},
        year={2001}
    }
"""
import torch
import logging
import numpy as np

from tqdm import tqdm
from typing import Tuple, List, Union


def suggested_projection_dim(dataset_size: int) -> int:
    """
        Returns the suggested projection dimensionality
        following the cited paper, which uses the formula
        floor(100 * log(N)), where N is the dataset size

        @article{roburin2022take,
            title={Take One Gram of Neural Features, Get Enhanced Group Robustness},
            author={Roburin, Simon and Corbi{\`e}re, Charles and Puy, Gilles and Thome, Nicolas and Aubry, Matthieu and Marlet, Renaud and P{\'e}rez, Patrick},
            journal={arXiv preprint arXiv:2208.12625},
            year={2022}
        }
    """
    return int(np.floor(100 * np.log(dataset_size)))


def suggested_scaling_factor(projection_dim: int) -> float:
    """
        Return the scaling factor for the projected
        matrix according to

        @article{roburin2022take,
            title={Take One Gram of Neural Features, Get Enhanced Group Robustness},
            author={Roburin, Simon and Corbi{\`e}re, Charles and Puy, Gilles and Thome, Nicolas and Aubry, Matthieu and Marlet, Renaud and P{\'e}rez, Patrick},
            journal={arXiv preprint arXiv:2208.12625},
            year={2022}
        }
    """
    return 1.0 / torch.sqrt(torch.tensor(projection_dim))


def generate_projection_matrix(dims: Tuple[int, int], device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """
        Generates a matrix filled with 1 or -1 with
        a 50% probability.

        :param dims:
            the dimensions of the projection matrix,
            given as (out_dim, in_dim)

        :returns:
            the projection matrix, on the given device
            and with a float32 data type
    """
    projection = ((torch.rand(dims) - 0.5) > 0).to(torch.int8).to(device)
    projection[projection == 0] = -1

    return projection.to(torch.float32)


def generate_autosized_projection_matrix(
    n_train_examples: int,
    grads_dimensionality: int,
    device: torch.device,
    dtype: torch.dtype = torch.float16
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
        Generates a projection matrix of size (floor(100 * log(N)), original_size)
        and returns it alongside the suggested scaling factor for projected
        vectors
    """
    projection_dim = suggested_projection_dim(dataset_size=n_train_examples)
    scaling_factor = suggested_scaling_factor(projection_dim=projection_dim)
    projection_matrix = generate_projection_matrix(
        dims=(projection_dim, grads_dimensionality),
        device=device
    ).to(dtype)

    return projection_matrix, scaling_factor


def project_dataset(
    projection: torch.Tensor,
    device: torch.device,
    dataset: np.ndarray,
    scaling_factor: float,
    normalize: bool = False,
    as_matrix: bool = False,
    indices: np.ndarray = None
) -> Union[List[np.ndarray], np.ndarray]:
    """
        Project the dataset using the given projection matrix and scaling
        factor. If specified, the projected vectors are also normalized.
    """
    features = []

    # Only project the given indices if any, otherwise
    # project the whole dataset
    if indices is None:
        indices = np.arange(dataset.shape[0])

    for i in tqdm(indices):
        sample = torch.tensor(
            dataset[i],
            dtype=projection.dtype,
            device=device
        ).view(-1)

        feature = scaling_factor * (projection @ sample)

        if normalize:
            feature = feature / torch.norm(feature, p=2)

        if torch.isnan(feature).sum() > 0 or torch.isinf(feature).sum() > 0:
            logging.error(f"found a NaN/inf value at {i}!")

        features.append(feature.cpu().numpy())

    if not as_matrix:
        return features
    else:
        # Return the data as a [num examples, projection dim] matrix
        return torch.cat([
            torch.tensor(x).unsqueeze(dim=0) for x in features
        ], dim=0).numpy()
