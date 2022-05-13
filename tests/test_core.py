"""Tests for utility functions"""

import numpy as np
import pandas as pd
import pytest

from multicons import (
    build_membership_matrix,
    cons_tree,
    in_ensemble_similarity,
    linear_closed_itemsets_miner,
    multicons,
)
from multicons.core import build_bi_clust, ensemble_jaccard_score


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
def test_build_membership_matrix_with_invalid_input(invalid):
    """Tests the build_memebership_matrix raises an error on invalid input."""

    error = "base_clusterings should contain at least one np.ndarray."
    with pytest.raises(IndexError, match=error):
        build_membership_matrix(invalid)


def test_in_ensemble_similarity():
    """Tests the in_ensemble_similarity should return the expected result."""

    base_clusterings = [np.array([0, 1, 1]), np.array([1, 0, 2]), np.array([0, 1, 2])]
    assert in_ensemble_similarity(base_clusterings) - 0.233 < 0.001
    base_clusterings = [np.array([0, 1]), np.array([0, 1])]
    assert in_ensemble_similarity(base_clusterings) == 1


@pytest.mark.parametrize("invalid", ["", None, [], [[1, 2, 3]]])
def test_in_ensemble_similarity_with_invalid_input(invalid):
    """Tests the in_ensemble_similarity raises an error on invalid input."""

    error = "base_clusterings should contain at least two np.ndarrays."
    with pytest.raises(IndexError, match=error):
        in_ensemble_similarity(invalid)


def test_build_bi_clust():
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


def test_multicons_without_any_parameters_using_jaccard_similarity():
    """Tests the multicons function without passing any optional parameters."""

    base_clusterings = [
        np.array([0, 0, 0, 1, 1, 1, 1, 1, 1]),
        np.array([0, 0, 0, 1, 1, 1, 1, 1, 1]),
        np.array([0, 0, 0, 0, 0, 1, 1, 2, 2]),
        np.array([1, 1, 1, 0, 0, 0, 0, 2, 2]),
        np.array([1, 1, 1, 0, 0, 0, 0, 2, 2]),
    ]
    value = multicons(base_clusterings)
    expected_consensus = np.array(
        [
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
            np.array([0, 0, 0, 1, 1, 1, 1, 1, 1]),
            np.array([0, 0, 0, 2, 2, 2, 2, 1, 1]),
            np.array([1, 1, 1, 0, 0, 2, 2, 3, 3]),
        ]
    )
    np.testing.assert_array_equal(value["consensus_vectors"], expected_consensus)
    expected_similarity = np.array([0.37777778, 0.67222222, 0.69722222, 0.47333333])
    similarity = value["ensemble_similarity"]
    assert (np.absolute(similarity - expected_similarity) < 0.0000001).all()
    assert value["recommended"] == 2
    assert value["tree_quality"] == 1
    np.testing.assert_array_equal(value["decision_thresholds"], np.array([1, 2, 4, 5]))
    np.testing.assert_array_equal(value["stability"], np.array([1, 1, 2, 1]))

    # Given a membership matrix and clusterings_count instead of base_clusterings
    # Should produce the same result.
    membership_matrix = build_membership_matrix(base_clusterings)
    membership_value = multicons(
        membership_matrix, is_membership_matrix=True, clusterings_count=5
    )
    np.testing.assert_array_equal(
        membership_value["consensus_vectors"], value["consensus_vectors"]
    )


def test_multicons_with_jaccard_index_similarity_measure():
    """Tests the multicons function with `similarity_measure="JaccardIndex"` and
    `optimize_label_names=True`.
    """

    base_clusterings = [
        np.array([0, 0, 0, 1, 1, 1, 1, 1, 1]),
        np.array([0, 0, 0, 1, 1, 1, 1, 1, 1]),
        np.array([0, 0, 0, 0, 0, 1, 1, 2, 2]),
        np.array([1, 1, 1, 0, 0, 0, 0, 2, 2]),
        np.array([1, 1, 1, 0, 0, 0, 0, 2, 2]),
    ]
    value = multicons(
        base_clusterings, similarity_measure="JaccardIndex", optimize_label_names=True
    )
    expected_consensus = np.array(
        [
            np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]),
            np.array([0, 0, 0, 1, 1, 1, 1, 1, 1]),
            np.array([0, 0, 0, 1, 1, 1, 1, 2, 2]),
            np.array([0, 0, 0, 3, 3, 1, 1, 2, 2]),
        ]
    )
    np.testing.assert_array_equal(value["consensus_vectors"], expected_consensus)
    expected_similarity = np.array([0.305, 0.47692308, 0.43181818, 0.33111888])
    similarity = value["ensemble_similarity"]
    assert (np.absolute(similarity - expected_similarity) < 0.0000001).all()
    assert value["recommended"] == 1
    assert value["tree_quality"] == 1
    np.testing.assert_array_equal(value["decision_thresholds"], np.array([1, 2, 4, 5]))
    np.testing.assert_array_equal(value["stability"], np.array([1, 1, 2, 1]))


def test_multicons_with_ensemble_jaccard_score_similarity_measure():
    """Tests the multicons function with `similarity_measure=ensemble_jaccard_score` and
    `optimize_label_names=True`.
    """

    base_clusterings = [
        np.array([0, 0, 0, 1, 1, 1, 1, 1, 1]),
        np.array([0, 0, 0, 1, 1, 1, 1, 1, 1]),
        np.array([0, 0, 0, 0, 0, 1, 1, 2, 2]),
        np.array([1, 1, 1, 0, 0, 0, 0, 2, 2]),
        np.array([1, 1, 1, 0, 0, 0, 0, 2, 2]),
    ]
    value = multicons(
        base_clusterings,
        similarity_measure=ensemble_jaccard_score,
        optimize_label_names=True,
    )
    expected_consensus = np.array(
        [
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
            np.array([0, 0, 0, 1, 1, 1, 1, 1, 1]),
            np.array([0, 0, 0, 2, 2, 2, 2, 1, 1]),
            np.array([1, 1, 1, 0, 0, 2, 2, 3, 3]),
        ]
    )
    expected_similarity = np.array([[0.55555556, 0.82666667, 0.80666667, 0.65]])
    assert value["recommended"] == 1
    np.testing.assert_array_equal(value["consensus_vectors"], expected_consensus)
    similarity = value["ensemble_similarity"]
    assert (np.absolute(similarity - expected_similarity) < 0.0000001).all()
    assert value["tree_quality"] == 1
    np.testing.assert_array_equal(value["decision_thresholds"], np.array([1, 2, 4, 5]))
    np.testing.assert_array_equal(value["stability"], np.array([1, 1, 2, 1]))


def test_cons_tree():
    """Tests the cons_tree function."""

    base_clusterings = [
        np.array([0, 0, 0, 1, 1, 1, 1, 1, 1]),
        np.array([0, 0, 0, 1, 1, 1, 1, 1, 1]),
        np.array([0, 0, 0, 0, 0, 1, 1, 2, 2]),
        np.array([1, 1, 1, 0, 0, 0, 0, 2, 2]),
        np.array([1, 1, 1, 0, 0, 0, 0, 2, 2]),
    ]
    tree = cons_tree(multicons(base_clusterings))
    assert str(tree).split("\n") == [
        "digraph {",
        '\tgraph [label="ConsTree',
        'Tree Quality = 1.0" labelloc=t]',
        "\tnode [fillcolor=slategray2 shape=ellipse style=filled width=9]",
        "\t00 [label=9]",
        "\tsubgraph cluster {",
        "\t\tgraph [label=Legend]",
        '\t\tnode [shape=box width=""]',
        '\t\tlegend_0 [label="DT=1 ST=1 Similarity=0.38"]',
        "\t}",
        "\tnode [fillcolor=slategray2 shape=ellipse style=filled width=3]",
        "\t10 [label=3]",
        "\t00 -> 10",
        "\tnode [fillcolor=slategray2 shape=ellipse style=filled width=6]",
        "\t11 [label=6]",
        "\t00 -> 11",
        "\tsubgraph cluster {",
        "\t\tgraph [label=Legend]",
        '\t\tnode [shape=box width=""]',
        '\t\tlegend_1 [label="DT=2 ST=1 Similarity=0.67"]',
        "\t\tlegend_0 -> legend_1",
        "\t}",
        "\tnode [fillcolor=darkseagreen shape=box style=filled width=3]",
        "\t20 [label=3]",
        "\t10 -> 20",
        "\tnode [fillcolor=darkseagreen shape=box style=filled width=2]",
        "\t21 [label=2]",
        "\t11 -> 21",
        "\tnode [fillcolor=darkseagreen shape=box style=filled width=4]",
        "\t22 [label=4]",
        "\t11 -> 22",
        "\tsubgraph cluster {",
        "\t\tgraph [label=Legend]",
        '\t\tnode [shape=box width=""]',
        '\t\tlegend_2 [label="DT=4 ST=2 Similarity=0.7"]',
        "\t\tlegend_1 -> legend_2",
        "\t}",
        "\tnode [fillcolor=slategray2 shape=ellipse style=filled width=2]",
        "\t30 [label=2]",
        "\t22 -> 30",
        "\tnode [fillcolor=slategray2 shape=ellipse style=filled width=3]",
        "\t31 [label=3]",
        "\t20 -> 31",
        "\tnode [fillcolor=slategray2 shape=ellipse style=filled width=2]",
        "\t32 [label=2]",
        "\t22 -> 32",
        "\tnode [fillcolor=slategray2 shape=ellipse style=filled width=2]",
        "\t33 [label=2]",
        "\t21 -> 33",
        "\tsubgraph cluster {",
        "\t\tgraph [label=Legend]",
        '\t\tnode [shape=box width=""]',
        '\t\tlegend_3 [label="DT=5 ST=1 Similarity=0.47"]',
        "\t\tlegend_2 -> legend_3",
        "\t}",
        "}",
        "",
    ]
