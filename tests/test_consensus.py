"""Tests for consensus functions"""

import pytest

from multicons import (
    consensus_function_10,
    consensus_function_12,
    consensus_function_13,
    consensus_function_14,
    consensus_function_15,
)


@pytest.mark.parametrize(
    "value, expected",
    [
        # A unique cluster remains unique
        ([{1}], [{1}]),
        # Clusters that are subsets are removed
        ([{1, 2}, {1}], [{1, 2}]),
        ([{1, 2}, {1, 2, 3}], [{1, 2, 3}]),
        # Clusters that are intersections are merged
        ([{1, 2, 3}, {2, 3, 4}, {5}], [{1, 2, 3, 4}, {5}]),
        ([{1, 2, 3}, {3, 4, 5}, {6}], [{1, 2, 3, 4, 5}, {6}]),
        ([{1, 2, 3, 4}, {4, 5, 6}, {7}], [{1, 2, 3, 4, 5, 6}, {7}]),
    ],
)
def test_consensus_consensus_function_10(value, expected):
    """Tests the consensus_function_10 should produce the expected value."""

    consensus_function_10(value)
    assert value == expected


@pytest.mark.parametrize(
    "value, expected",
    [
        # A unique cluster remains unique
        ([{1}], [{1}]),
        # Clusters that are subsets are removed
        ([{1, 2}, {1}], [{1, 2}]),
        ([{1, 2}, {1, 2, 3}], [{1, 2, 3}]),
        # Clusters that are intersections with a ratio higher than MT are merged
        ([{1, 2, 3}, {2, 3, 4}, {5}], [{1, 2, 3, 4}, {5}]),
        # Clusters that are intersections with a ration lower than MT are split
        ([{1, 2, 3}, {3, 4, 5}, {6}], [{1, 2, 3}, {4, 5}, {6}]),
        ([{1, 2, 3, 4}, {4, 5, 6}, {7}], [{1, 2, 3}, {4, 5, 6}, {7}]),
    ],
)
def test_consensus_consensus_function_12(value, expected):
    """Tests the consensus_function_12 should produce the expected value."""

    consensus_function_12(value)
    assert value == expected


@pytest.mark.parametrize(
    "value, expected",
    [
        # A unique cluster remains unique
        ([{1}], [{1}]),
        # Clusters that are subsets are removed
        ([{1, 2}, {1}], [{1, 2}]),
        ([{1, 2}, {1, 2, 3}], [{1, 2, 3}]),
        # Clusters that are intersections with a ratio higher than MT are merged
        ([{1, 2, 3}, {2, 3, 4}, {5}], [{1, 2, 3, 4}, {5}]),
        # Clusters that are intersections with a ration lower than MT are split
        ([{1, 2, 3}, {3, 4, 5}, {6}], [{1, 2}, {3, 4, 5}, {6}]),
        ([{1, 2, 3, 4}, {4, 5, 6}, {7}], [{1, 2, 3}, {4, 5, 6}, {7}]),
    ],
)
def test_consensus_consensus_function_13(value, expected):
    """Tests the consensus_function_13 should produce the expected value."""

    consensus_function_13(value)
    assert value == expected


@pytest.mark.parametrize(
    "value, expected",
    [
        # A unique cluster remains unique
        ([{1}], [{1}]),
        # Clusters that are subsets are removed
        ([{1, 2}, {1}], [{1, 2}]),
        ([{1, 2}, {1, 2, 3}], [{1, 2, 3}]),
        # Clusters that are intersections with a ratio higher than MT are merged
        ([{1, 2, 3}, {2, 3, 4}, {5}], [{1, 2, 3, 4}, {5}]),
        # Clusters that are intersections with a ration lower than MT are split
        ([{1, 2, 3}, {3, 4, 5}, {6}], [{1, 2, 3}, {4, 5}, {6}]),
        ([{1, 2, 3, 4}, {4, 5, 6}, {7}], [{1, 2, 3}, {4, 5, 6}, {7}]),
    ],
)
def test_consensus_consensus_function_14(value, expected):
    """Tests the consensus_function_14 should produce the expected value."""

    consensus_function_14(value)
    assert value == expected


@pytest.mark.parametrize(
    "value, expected",
    [
        # A unique cluster remains unique
        ([{1}], [{1}]),
        # Clusters that are subsets are removed
        ([{1, 2}, {1}], [{1, 2}]),
        ([{1, 2}, {1, 2, 3}], [{1, 2, 3}]),
        # Clusters that are intersections with a ratio higher than MT are merged
        ([{1, 2, 3}, {2, 3, 4}, {5}], [{1, 2, 3, 4}, {5}]),
        # Clusters that are intersections with a ration lower than MT are split
        ([{1, 2, 3}, {3, 4, 5}, {6}], [{1, 2, 3}, {4, 5}, {6}]),
        ([{1, 2, 3, 4}, {4, 5, 6}, {7}], [{1, 2, 3}, {4, 5, 6}, {7}]),
    ],
)
def test_consensus_consensus_function_15(value, expected):
    """Tests the consensus_function_15 should produce the expected value."""
    consensus_function_15(value)
    assert value == expected
