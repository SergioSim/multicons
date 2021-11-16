"""Utility functions"""

import os
import subprocess  # nosec
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory

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


def read_all_lines_from_directory(directory):
    """Yields all lines from all files present in directory."""

    for path in Path(directory).iterdir():
        with path.open(mode="r") as file:
            for line in file:
                yield line


def linear_closed_itemsets_miner(membership_matrix: pd.DataFrame):
    """Returns a list of frequent closed itemsets using the LCM algorithm."""

    # Thanks to https://github.com/slide-lig/plcmpp for the implementation
    # of the LCM algorithm! It's cloned and build in the ./src/plcmpp directory.
    path = None
    groups = list(range(0, len(membership_matrix.columns)))
    with NamedTemporaryFile(mode="w", delete=False) as file:
        path = file.name
        for i in range(0, len(membership_matrix)):
            transaction = (membership_matrix.iloc[i, :] * groups)[
                membership_matrix.iloc[i, :].astype(bool)
            ]
            file.write(" ".join(transaction.astype(str)) + "\n")
    frequent_closed_itemsets = []
    with TemporaryDirectory() as temp_dir:
        plcmpp = __file__.replace("utils.py", "") + "../plcmpp/src/pLCM++"
        subprocess.run([plcmpp, path, "0", "."], cwd=temp_dir, check=True)  # nosec
        for line in read_all_lines_from_directory(temp_dir):
            itemset = np.array(line.split("\t")[1].split(" ")).astype(int)
            itemset.sort()
            frequent_closed_itemsets.append(list(membership_matrix.columns[itemset]))
    os.unlink(path)
    frequent_closed_itemsets.sort(key=lambda x: (len(x), x), reverse=True)
    return frequent_closed_itemsets
