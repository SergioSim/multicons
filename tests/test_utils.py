"""Tests for utility functions"""

import numpy as np
import pandas as pd
import pytest

from multicons import build_membership_matrix, in_ensemble_similarity


def test_build_membership_matrix():
    """Tests the build_memebership_matrix should return the expected result."""

    base_clusterings = [np.array([0, 1, 1]), np.array([1, 0, 0]), np.array([0, 1, 2])]
    value = build_membership_matrix(base_clusterings)
    expected = pd.DataFrame(
        [
            [1, 0, 0, 1, 1, 0, 0],
            [0, 1, 1, 0, 0, 1, 0],
            [0, 1, 1, 0, 0, 0, 1],
        ],
        columns=["0P0", "0P1", "1P0", "1P1", "2P0", "2P1", "2P2"],
    )
    pd.testing.assert_frame_equal(value, expected)


@pytest.mark.parametrize("invalid", ["", None, [], [[1, 2, 3]]])
def test_build_membership_matrix_with_invalid_input(invalid):
    """Tests the build_memebership_matrix raises an error on invalid input."""

    error = "base_clusterings should contain at least one np.ndarray."
    with pytest.raises(IndexError, match=error):
        build_membership_matrix(invalid)


def test_in_ensemble_similarity():
    """Tests the in_ensemble_similarity should return the expected result."""

    base_clusterings = [np.array([0, 1, 1]), np.array([1, 0, 0]), np.array([0, 1, 2])]
    assert in_ensemble_similarity(base_clusterings) == 2 / 3


@pytest.mark.parametrize("invalid", ["", None, [], [[1, 2, 3]]])
def test_in_ensemble_similarity_with_invalid_input(invalid):
    """Tests the in_ensemble_similarity raises an error on invalid input."""

    error = "base_clusterings should contain at least one np.ndarray."
    with pytest.raises(IndexError, match=error):
        in_ensemble_similarity(invalid)
