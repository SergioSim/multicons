"""Utility functions"""

import numpy as np
import pandas as pd
from fim import eclat  # pylint: disable=no-name-in-module


def jaccard_index(predictions: np.ndarray, labels: np.ndarray):
    """Returns the jaccard index using the formula: |X ∩ Y| / (|X| + |Y| - |X ∩ Y|)."""

    intersection_size = (predictions == labels).sum()
    return intersection_size / (len(labels) + len(predictions) - intersection_size)


def jaccard_similarity(partition_a: np.ndarray, partition_b: np.ndarray):
    """Returns the jaccard similarity score using the formula: yy / (yy + ny + yn)."""

    partition_a_matrix = np.transpose(np.meshgrid(partition_a, partition_a), (2, 1, 0))
    partition_b_matrix = np.transpose(np.meshgrid(partition_b, partition_b), (2, 1, 0))
    ai_in_aj = partition_a_matrix[:, :, 0] == partition_a_matrix[:, :, 1]
    bi_in_bj = partition_b_matrix[:, :, 0] == partition_b_matrix[:, :, 1]
    ab_yy = np.sum(ai_in_aj & bi_in_bj) - len(partition_a)
    ab_yn_ny = np.sum((~ai_in_aj & bi_in_bj) | (ai_in_aj & ~bi_in_bj))
    return ab_yy / (ab_yy + ab_yn_ny)


def ensemble_jaccard_score(predictions: np.ndarray, labels: np.ndarray):
    """Computes a jaccard score without considering the underlining labels."""

    predictions = [set(np.where(predictions == x)[0]) for x in np.unique(predictions)]
    labels = [set(np.where(labels == x)[0]) for x in np.unique(labels)]
    score = 0
    for prediction in predictions:
        intersections = []
        for label in labels:
            intersections.append(
                len(prediction.intersection(label)) / len(prediction.union(label))
            )
        score += max(intersections)
    return score / len(predictions)


def in_ensemble_similarity(base_clusterings: list[np.ndarray]) -> float:
    """Returns the average similarity among the base clusters using Jaccard score."""

    if not base_clusterings or len(base_clusterings) < 2:
        raise IndexError("base_clusterings should contain at least two np.ndarrays.")

    count = len(base_clusterings)
    index = np.arange(count)
    similarity = pd.DataFrame(0.0, index=index, columns=index)
    average_similarity = np.zeros(count)
    for i in range(count - 1):
        cluster_i = base_clusterings[i]
        for j in range(i + 1, count):
            cluster_j = base_clusterings[j]
            score = jaccard_index(cluster_i, cluster_j)
            similarity.iloc[i, j] = similarity.iloc[j, i] = score
        average_similarity[i] = similarity.iloc[i].sum() / (count - 1)

    average_similarity[count - 1] = similarity.iloc[count - 1].sum() / (count - 1)
    return np.mean(average_similarity)


def build_membership_matrix(base_clusterings: np.ndarray) -> pd.DataFrame:
    """Computes and returns the membership matrix."""

    if len(base_clusterings) == 0 or not isinstance(base_clusterings[0], np.ndarray):
        raise IndexError("base_clusterings should contain at least one np.ndarray.")

    res = []
    for clusters in base_clusterings:
        res += [clusters == x for x in np.unique(clusters)]
    return pd.DataFrame(np.transpose(res), dtype=bool)


def build_base_clusterings(membership_matrix: pd.DataFrame) -> np.ndarray:
    """Computes and returns the base_clusterings."""

    clusters_count = (membership_matrix.iloc[0]).sum()
    cluster_size = membership_matrix.shape[0]
    base_clusterings = np.zeros((clusters_count, cluster_size), dtype=int)
    column = 0
    for i in range(clusters_count):
        items_count = 0
        label = 0
        while items_count < cluster_size:
            items = np.nonzero(membership_matrix.iloc[:, column].values)[0]
            base_clusterings[i][items] = label
            items_count += items.size
            column += 1
            label += 1
    return base_clusterings


def build_bi_clust(
    membership_matrix: pd.DataFrame,
    frequent_closed_itemsets: list[frozenset],
    size: int,
) -> list[set]:
    """Returns a new list of instances from itemsets of given size."""

    itemsets = list(filter(lambda x: len(x) == size, frequent_closed_itemsets))
    result = []
    for itemset in itemsets:
        consensus = np.ones(len(membership_matrix), dtype=bool)
        for partition in itemset:
            consensus = consensus & membership_matrix.iloc[:, partition]
        result.append(set(consensus[consensus].index.values))
    result.sort(key=len)
    return result


def linear_closed_itemsets_miner(membership_matrix: pd.DataFrame):
    """Returns a list of frequent closed itemsets using the LCM algorithm."""

    transactions = []
    for i in membership_matrix.index:
        transactions.append(np.nonzero(membership_matrix.iloc[i].values)[0].tolist())
    frequent_closed_itemsets = eclat(transactions, target="c", supp=0, algo="o", conf=0)
    return sorted(map(lambda x: frozenset(x[0]), frequent_closed_itemsets), key=len)
