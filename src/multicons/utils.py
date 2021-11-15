"""Utility functions"""

import numpy as np
import pandas as pd


def build_membership_matrix(base_clusterings: list[np.ndarray]) -> pd.DataFrame:
    """Computes and returns the membership matrix."""

    if not base_clusterings or not isinstance(base_clusterings[0], np.ndarray):
        raise IndexError("base_clusterings should contain at least one np.ndarray.")

    index = range(0, base_clusterings[0].size)
    membership_matrix = pd.DataFrame(index=index)
    for i, clusters in enumerate(base_clusterings):
        columns = [f"{i}P{partition}" for partition in np.unique(clusters)]
        values = np.eye(len(columns), dtype=int)[clusters]
        membership_matrix[columns] = pd.DataFrame(values, index=index)
    return membership_matrix
