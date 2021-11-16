"""Tests for utility functions"""

from subprocess import CalledProcessError

import numpy as np
import pandas as pd
import pytest

from multicons import (
    build_membership_matrix,
    in_ensemble_similarity,
    linear_closed_itemsets_miner,
)


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


def test_linear_closed_itemsets_miner():
    """Tests the linear_closed_itemsets_miner function."""

    membership_matrix = pd.DataFrame(
        [
            [1, 0, 0, 1, 1, 0, 0],
            [0, 1, 1, 0, 0, 1, 0],
            [0, 1, 1, 0, 0, 0, 1],
        ],
        columns=["0P0", "0P1", "1P0", "1P1", "2P0", "2P1", "2P2"],
    )
    expected = [
        ["0P1", "1P0", "2P2"],
        ["0P1", "1P0", "2P1"],
        ["0P0", "1P1", "2P0"],
        ["0P1", "1P0"],
    ]
    assert linear_closed_itemsets_miner(membership_matrix) == expected

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
        columns=[
            "1P1",
            "1P2",
            "2P1",
            "2P2",
            "3P1",
            "3P2",
            "3P3",
            "4P1",
            "4P2",
            "4P3",
            "5P1",
            "5P2",
            "5P3",
        ],
    )
    expected = [
        ["1P2", "2P2", "3P3", "4P3", "5P3"],
        ["1P2", "2P2", "3P2", "4P1", "5P1"],
        ["1P2", "2P2", "3P1", "4P1", "5P1"],
        ["1P1", "2P1", "3P1", "4P2", "5P2"],
        ["1P2", "2P2", "4P1", "5P1"],
        ["1P2", "2P2"],
        ["3P1"],
    ]
    assert linear_closed_itemsets_miner(membership_matrix) == expected


def test_linear_closed_itemsets_miner_with_invalid_input():
    """Tests the in_ensemble_similarity raises an error on invalid input."""

    invalid = pd.DataFrame(["test", "124", "hello"])
    error = "Command '.*' died with <Signals.SIGILL: 4>"
    with pytest.raises(CalledProcessError, match=error):
        linear_closed_itemsets_miner(invalid)
