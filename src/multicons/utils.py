"""Utility functions"""

import numpy as np
import pandas as pd
from sklearn.metrics import jaccard_score


def build_membership_matrix(base_clusterings: list[np.ndarray]) -> pd.DataFrame:
    """Computes and returns the membership matrix."""

    if not base_clusterings or not isinstance(base_clusterings[0], np.ndarray):
        raise IndexError("base_clusterings should contain at least one np.ndarray.")

    index = np.arange(0, base_clusterings[0].size)
    membership_matrix = pd.DataFrame(index=index)
    for i, clusters in enumerate(base_clusterings):
        columns = [f"{i}P{partition}" for partition in np.unique(clusters)]
        values = np.eye(len(columns), dtype=int)[clusters]
        membership_matrix[columns] = pd.DataFrame(values, index=index)
    return membership_matrix


def in_ensemble_similarity(base_clusterings: list[np.ndarray]) -> float:
    """Returns the average similarity among the base clusters using Jaccard score."""

    if not base_clusterings or not isinstance(base_clusterings[0], np.ndarray):
        raise IndexError("base_clusterings should contain at least one np.ndarray.")

    count = len(base_clusterings)
    index = np.arange(0, count)
    similarity = pd.DataFrame(0.0, index=index, columns=index)
    average_similarity = np.zeros(count)
    for i in range(0, count - 1):
        cluster_i = base_clusterings[i]
        for j in range(i + 1, count):
            cluster_j = base_clusterings[j]
            similarity[i, j] = similarity[j, i] = jaccard_score(
                cluster_i, cluster_j, average="weighted"
            )
        average_similarity[i] = similarity.iloc[i].sum() / (count - 1)
    average_similarity[count - 1] = similarity.iloc[count - 1].sum() / (count - 1)
    return np.mean(average_similarity)
