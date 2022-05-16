"""Tests for consensus functions"""

import pytest

from multicons import consensus_function_10


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
    ],
)
def test_consensus_consensus_function_10(value, expected):
    """Tests the consensus_function_10 should produce the expected value."""

    consensus_function_10(value)
    assert value == expected
