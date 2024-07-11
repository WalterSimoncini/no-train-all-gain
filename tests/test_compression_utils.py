import torch
import numpy as np

from src.utils import get_device
from src.utils.compression import (
    generate_projection_matrix,
    generate_autosized_projection_matrix
)


def test_generate_projection_matrix():
    """
        Make sure that the generated matrix has
        the right shape, type and values
    """
    matrix = generate_projection_matrix((30, 100))

    assert type(matrix) == torch.Tensor
    assert matrix.shape == torch.Size([30, 100])
    assert matrix.dtype == torch.float32
    assert set(matrix.unique().tolist()) == set([1, -1])


def test_autosized_dtype():
    """
        Make sure that generate_autosized_projection_matrix
        generates a matrix with appropriate dimensions and
        data type
    """
    n_examples = 10000
    grads_dimensionality = 512
    device = get_device()

    projection, _ = generate_autosized_projection_matrix(
        n_train_examples=n_examples,
        grads_dimensionality=grads_dimensionality,
        device=device
    )

    assert projection.shape == torch.Size([
        int(np.floor(np.log(n_examples) * 100).item()),
        grads_dimensionality
    ])

    # Make sure the default data type is float16
    assert projection.dtype == torch.float16

    # Make sure the data type is the one
    # specified by the user
    projection, _ = generate_autosized_projection_matrix(
        n_train_examples=n_examples,
        grads_dimensionality=grads_dimensionality,
        device=device,
        dtype=torch.float32
    )

    assert projection.dtype == torch.float32
