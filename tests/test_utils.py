"""Tests for utility functions"""

import numpy as np
import pandas as pd
import pytest

from multicons import (
    build_membership_matrix,
    in_ensemble_similarity,
    linear_closed_itemsets_miner,
)
from multicons.utils import build_bi_clust


def test_utils_build_membership_matrix():
    """Tests the build_memebership_matrix should return the expected result."""

    base_clusterings = [np.array([0, 1, 1]), np.array([1, 0, 0]), np.array([0, 1, 2])]
    value = build_membership_matrix(base_clusterings)
    expected = pd.DataFrame(
        [
            [1, 0, 0, 1, 1, 0, 0],
            [0, 1, 1, 0, 0, 1, 0],
            [0, 1, 1, 0, 0, 0, 1],
        ],
        columns=list(range(7)),
    )
    pd.testing.assert_frame_equal(value, expected)

    base_clusterings = [
        np.array([0, 0, 0, 1, 1, 1, 1, 1, 1]),
        np.array([0, 0, 0, 1, 1, 1, 1, 1, 1]),
        np.array([0, 0, 0, 0, 0, 1, 1, 2, 2]),
        np.array([1, 1, 1, 0, 0, 0, 0, 2, 2]),
        np.array([1, 1, 1, 0, 0, 0, 0, 2, 2]),
    ]
    value = build_membership_matrix(base_clusterings)
    expected = pd.DataFrame(
        [
            [1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
            [1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
            [1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
            [0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0],
            [0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0],
            [0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
            [0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
        ],
        columns=list(range(13)),
    )
    pd.testing.assert_frame_equal(value, expected)


@pytest.mark.parametrize("invalid", [[], [[1, 2, 3]]])
def test_utils_build_membership_matrix_with_invalid_input(invalid):
    """Tests the build_memebership_matrix raises an error on invalid input."""

    error = "base_clusterings should contain at least one np.ndarray."
    with pytest.raises(IndexError, match=error):
        build_membership_matrix(invalid)


def test_utils_in_ensemble_similarity():
    """Tests the in_ensemble_similarity should return the expected result."""

    base_clusterings = [np.array([0, 1, 1]), np.array([1, 0, 2]), np.array([0, 1, 2])]
    assert in_ensemble_similarity(base_clusterings) - 0.233 < 0.001
    base_clusterings = [np.array([0, 1]), np.array([0, 1])]
    assert in_ensemble_similarity(base_clusterings) == 1


@pytest.mark.parametrize("invalid", ["", None, [], [[1, 2, 3]]])
def test_utils_in_ensemble_similarity_with_invalid_input(invalid):
    """Tests the in_ensemble_similarity raises an error on invalid input."""

    error = "base_clusterings should contain at least two np.ndarrays."
    with pytest.raises(IndexError, match=error):
        in_ensemble_similarity(invalid)


def test_utils_build_bi_clust():
    """Tests the build_bi_clust function."""

    membership_matrix = pd.DataFrame(
        [
            [1, 0, 0, 1],
            [1, 0, 1, 0],
            [0, 1, 1, 0],
            [0, 1, 1, 0],
        ],
        columns=["A", "B", "C", "D"],
    )
    frequent_closed_itemsets = [
        frozenset(["A", "C"]),
        frozenset(["B", "C"]),
        frozenset(["A", "D"]),
        frozenset(["A", "B", "C"]),
    ]
    assert build_bi_clust(membership_matrix, frequent_closed_itemsets, 2) == [
        set([1]),
        set([2, 3]),
        set([0]),
    ]


def test_utils_linear_closed_itemsets_miner():
    """Tests the linear_closed_itemsets_miner function."""

    membership_matrix = pd.DataFrame(
        [
            [1, 0, 0, 1, 1, 0, 0],
            [0, 1, 1, 0, 0, 1, 0],
            [0, 1, 1, 0, 0, 0, 1],
        ],
        columns=["0P0", "0P1", "1P0", "1P1", "2P0", "2P1", "2P2"],
    )
    expected = {
        frozenset([1, 2]),
        frozenset([0, 3, 4]),
        frozenset([1, 2, 5]),
        frozenset([1, 2, 6]),
    }
    assert set(linear_closed_itemsets_miner(membership_matrix)) == expected

    membership_matrix = pd.DataFrame(
        [
            [1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
            [1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
            [1, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0],
            [0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0],
            [0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 1, 0, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0],
            [0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0],
            [0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
            [0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1],
        ],
        columns=list(range(13)),
    )
    expected = [
        frozenset([4]),
        frozenset([1, 3]),
        frozenset([1, 3, 7, 10]),
        frozenset([1, 3, 4, 7, 10]),
        frozenset([0, 2, 4, 8, 11]),
        frozenset([1, 3, 5, 7, 10]),
        frozenset([1, 3, 6, 9, 12]),
    ]
    assert linear_closed_itemsets_miner(membership_matrix) == expected
