import pytest
import numpy as np

from src.utils.metrics import mean_per_class_accuracy


def test_mean_per_class_accuracy():
    targets = np.array([1, 2, 3, 4])

    # No single correct predictions. The score should be zero
    assert mean_per_class_accuracy(
        np.roll(targets, 1),
        targets
    ) == 0

    # Remove the second class. As the second class does
    # not exist, this should not cause issues with the
    # denominator being zero, as it does not exist
    targets = np.array([1, 1, 3, 4])
    preds = np.array([1, 1, 4, 3])

    # Three classes, one fully correct and the other two wrong.
    assert round(mean_per_class_accuracy(
        preds,
        targets
    ), 2) == pytest.approx(0.33, 0.01)

    # Make sure we have a perfect score if all
    # predictions are correct
    assert mean_per_class_accuracy(
        targets,
        targets
    ) == pytest.approx(1.0, 0.01)

    targets = np.array([1, 2, 2, 1])
    preds = np.array([1, 1, 2, 2])

    # Two classes, both at 50% accuracy
    assert mean_per_class_accuracy(
        targets,
        preds
    ) == pytest.approx(0.5, 0.01)
