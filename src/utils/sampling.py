import random
import numpy as np

from collections import defaultdict


def sample_balanced(targets: np.ndarray, n_examples_per_class: int, shuffle: bool = False) -> np.ndarray:
    """
        Returns the indices of the targets array that make up a
        balanced dataset where each class is represented by N examples
    """
    # Create a target --> indices mapping
    target2indices = defaultdict(list)

    for i, target in enumerate(targets):
        target2indices[target].append(i)

    # For each class label, select the top-k indices
    selected_indices = []

    for key, value in target2indices.items():
        if shuffle:
            random.shuffle(value)

        if len(value) < n_examples_per_class:
            raise ValueError(f"class {key} has only {len(value)} examples, while {n_examples_per_class} should have been sampled")

        selected_indices.extend(value[:n_examples_per_class])

    # Return the selected indices. The array should have shape n_examples_per_class * n_classes
    return np.array(selected_indices)
