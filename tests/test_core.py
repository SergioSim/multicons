"""Tests for core functions"""

import numpy as np
import pytest

from multicons import MultiCons, build_membership_matrix
from multicons.utils import ensemble_jaccard_score


def test_core_multicons_with_invalid_parameters():
    """Tests that the MultiCons class raises an Exception given invalid parameters."""

    with pytest.raises(ValueError, match="Invalid value"):
        MultiCons(similarity_measure="foo")


def test_core_multicons_without_any_parameters_using_jaccard_similarity():
    """Tests the MultiCons class without passing any optional parameters."""

    base_clusterings = [
        np.array([0, 0, 0, 1, 1, 1, 1, 1, 1]),
        np.array([0, 0, 0, 1, 1, 1, 1, 1, 1]),
        np.array([0, 0, 0, 0, 0, 1, 1, 2, 2]),
        np.array([1, 1, 1, 0, 0, 0, 0, 2, 2]),
        np.array([1, 1, 1, 0, 0, 0, 0, 2, 2]),
    ]
    value = MultiCons().fit(base_clusterings)
    expected_consensus = np.array(
        [
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
            np.array([0, 0, 0, 1, 1, 1, 1, 1, 1]),
            np.array([1, 1, 1, 2, 2, 2, 2, 0, 0]),
            np.array([3, 3, 3, 0, 0, 1, 1, 2, 2]),
        ]
    )
    np.testing.assert_array_equal(value.consensus_vectors, expected_consensus)
    expected_similarity = np.array([0.37777778, 0.67222222, 0.69722222, 0.47333333])
    assert (np.absolute(value.ensemble_similarity - expected_similarity) < 0.0001).all()
    assert value.recommended == 2
    assert value.tree_quality == 1
    np.testing.assert_array_equal(value.decision_thresholds, np.array([1, 2, 4, 5]))
    np.testing.assert_array_equal(value.stability, np.array([1, 1, 2, 1]))

    # Given a membership matrix and clusterings_count instead of base_clusterings
    # Should produce the same result.
    membership_matrix = build_membership_matrix(base_clusterings)
    membership_value = MultiCons().fit(membership_matrix)
    np.testing.assert_array_equal(
        membership_value.consensus_vectors, value.consensus_vectors
    )
    np.testing.assert_array_equal(
        membership_value.ensemble_similarity, value.ensemble_similarity
    )


def test_core_multicons_with_jaccard_index_similarity_measure():
    """Tests the MultiCons class with `similarity_measure="JaccardIndex"` and
    `optimize_label_names=True`.
    """

    base_clusterings = [
        np.array([0, 0, 0, 1, 1, 1, 1, 1, 1]),
        np.array([0, 0, 0, 1, 1, 1, 1, 1, 1]),
        np.array([0, 0, 0, 0, 0, 1, 1, 2, 2]),
        np.array([1, 1, 1, 0, 0, 0, 0, 2, 2]),
        np.array([1, 1, 1, 0, 0, 0, 0, 2, 2]),
    ]
    value = MultiCons(similarity_measure="JaccardIndex", optimize_label_names=True).fit(
        base_clusterings
    )
    expected_consensus = np.array(
        [
            np.array([1, 1, 1, 1, 1, 1, 1, 1, 1]),
            np.array([0, 0, 0, 1, 1, 1, 1, 1, 1]),
            np.array([0, 0, 0, 1, 1, 1, 1, 2, 2]),
            np.array([0, 0, 0, 3, 3, 1, 1, 2, 2]),
        ]
    )
    np.testing.assert_array_equal(value.consensus_vectors, expected_consensus)
    expected_similarity = np.array([0.305, 0.47692308, 0.43181818, 0.33111888])
    assert (np.absolute(value.ensemble_similarity - expected_similarity) < 0.0001).all()
    assert value.recommended == 1
    assert value.tree_quality == 1
    np.testing.assert_array_equal(value.decision_thresholds, np.array([1, 2, 4, 5]))
    np.testing.assert_array_equal(value.stability, np.array([1, 1, 2, 1]))


def test_core_multicons_with_ensemble_jaccard_score_similarity_measure():
    """Tests the MultiCons class with `similarity_measure=ensemble_jaccard_score` and
    `optimize_label_names=True`.
    """

    base_clusterings = [
        np.array([0, 0, 0, 1, 1, 1, 1, 1, 1]),
        np.array([0, 0, 0, 1, 1, 1, 1, 1, 1]),
        np.array([0, 0, 0, 0, 0, 1, 1, 2, 2]),
        np.array([1, 1, 1, 0, 0, 0, 0, 2, 2]),
        np.array([1, 1, 1, 0, 0, 0, 0, 2, 2]),
    ]
    value = MultiCons(
        similarity_measure=ensemble_jaccard_score,
        optimize_label_names=True,
    ).fit(base_clusterings)
    expected_consensus = np.array(
        [
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
            np.array([0, 0, 0, 1, 1, 1, 1, 1, 1]),
            np.array([1, 1, 1, 2, 2, 2, 2, 0, 0]),
            np.array([3, 3, 3, 0, 0, 1, 1, 2, 2]),
        ]
    )
    expected_similarity = np.array([[0.55555556, 0.82666667, 0.80666667, 0.65]])
    assert value.recommended == 1
    np.testing.assert_array_equal(value.consensus_vectors, expected_consensus)
    assert (np.absolute(value.ensemble_similarity - expected_similarity) < 0.0001).all()
    assert value.tree_quality == 1
    np.testing.assert_array_equal(value.decision_thresholds, np.array([1, 2, 4, 5]))
    np.testing.assert_array_equal(value.stability, np.array([1, 1, 2, 1]))


def test_core_multicons_with_consensus_function_12():
    """Tests the MultiCons class using the `consensus_function_12`."""

    base_clusterings = [
        np.array([1, 1, 1, 1, 2, 2, 2, 2, 2]),
        np.array([1, 1, 1, 2, 2, 2, 2, 2, 2]),
        np.array([1, 1, 1, 1, 1, 2, 2, 2, 2]),
        np.array([2, 2, 2, 2, 2, 1, 1, 2, 2]),
        np.array([2, 2, 1, 1, 1, 1, 1, 1, 1]),
    ]
    value = MultiCons(consensus_function="consensus_function_12").fit(base_clusterings)
    expected_consensus = np.array(
        [
            np.array([0, 0, 0, 0, 0, 0, 0, 0, 0]),
            np.array([1, 1, 0, 0, 0, 2, 2, 2, 2]),
            np.array([4, 4, 2, 1, 0, 5, 5, 3, 3]),
        ]
    )
    np.testing.assert_array_equal(value.consensus_vectors, expected_consensus)


def test_core_multicons_cons_tree():
    """Tests the MultiCons.cons_tree method."""

    base_clusterings = [
        np.array([0, 0, 0, 1, 1, 1, 1, 1, 1]),
        np.array([0, 0, 0, 1, 1, 1, 1, 1, 1]),
        np.array([0, 0, 0, 0, 0, 1, 1, 2, 2]),
        np.array([1, 1, 1, 0, 0, 0, 0, 2, 2]),
        np.array([1, 1, 1, 0, 0, 0, 0, 2, 2]),
    ]
    tree = MultiCons().fit(base_clusterings).cons_tree()
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
        "\tnode [fillcolor=darkseagreen shape=box style=filled width=2]",
        "\t20 [label=2]",
        "\t11 -> 20",
        "\tnode [fillcolor=darkseagreen shape=box style=filled width=3]",
        "\t21 [label=3]",
        "\t10 -> 21",
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
        "\tnode [fillcolor=slategray2 shape=ellipse style=filled width=2]",
        "\t31 [label=2]",
        "\t22 -> 31",
        "\tnode [fillcolor=slategray2 shape=ellipse style=filled width=2]",
        "\t32 [label=2]",
        "\t20 -> 32",
        "\tnode [fillcolor=slategray2 shape=ellipse style=filled width=3]",
        "\t33 [label=3]",
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
