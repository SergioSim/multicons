"""Consensus functions definitions."""

# pylint: disable=unused-argument

import pandas as pd
import scipy.sparse as sp


def consensus_function_10(bi_clust: list[set], merging_threshold=None):
    """Returns a modified bi_clust (set of unique instance sets)."""

    i = 0
    count = len(bi_clust)
    while i < count - 1:
        bi_clust_i = bi_clust[i]
        j = i + 1
        while j < count:
            bi_clust_j = bi_clust[j]
            intersection_size = len(bi_clust_i.intersection(bi_clust_j))
            if intersection_size == 0:
                j += 1
                continue
            if intersection_size == len(bi_clust_i):
                # Bi⊂Bj
                del bi_clust[i]
                count -= 1
                i -= 1
                break
            if intersection_size == len(bi_clust_j):
                # Bj⊂Bi
                del bi_clust[j]
                count -= 1
                continue
            bi_clust[j] = bi_clust_i.union(bi_clust_j)
            del bi_clust[i]
            count -= 1
            i -= 1
            break
        i += 1


def consensus_function_12(bi_clust: list[set], merging_threshold: float = 0.5):
    """Returns a modified bi_clust (set of unique instance sets)."""

    i = 0
    count = len(bi_clust)
    while i < count - 1:
        bi_clust_i = bi_clust[i]
        bi_clust_i_size = len(bi_clust_i)
        j = i + 1
        while j < count:
            bi_clust_j = bi_clust[j]
            bi_clust_j_size = len(bi_clust_j)
            bi_clust_intersection = bi_clust_i.intersection(bi_clust_j)
            intersection_size = len(bi_clust_intersection)
            if intersection_size == 0:
                j += 1
                continue
            if intersection_size == bi_clust_i_size:
                # Bi⊂Bj
                del bi_clust[i]
                count -= 1
                i -= 1
                break
            if intersection_size == bi_clust_j_size:
                # Bj⊂Bi
                del bi_clust[j]
                count -= 1
                continue
            if (
                intersection_size >= bi_clust_i_size * merging_threshold
                or intersection_size >= bi_clust_j_size * merging_threshold
            ):
                # Merge intersecting sets (Bj∩Bi / |Bi| > MT or Bj∩Bi / |Bj| > MT)
                bi_clust[j] = bi_clust_i.union(bi_clust_j)
                del bi_clust[i]
                count -= 1
                i -= 1
                break
            # Split intersecting sets (remove intersection from bigger set)
            if bi_clust_i_size <= bi_clust_j_size:
                bi_clust[j] = bi_clust_j - bi_clust_intersection
                continue
            bi_clust[i] = bi_clust_i - bi_clust_intersection
            i -= 1
            break
        i += 1


def consensus_function_13(bi_clust: list[set], merging_threshold: float = 0.5):
    """Returns a modified bi_clust (set of unique instance sets)."""

    i = 0
    count = len(bi_clust)
    merging_threshold *= 2
    while i < count - 1:
        bi_clust_i = bi_clust[i]
        bi_clust_i_size = len(bi_clust_i)
        j = i + 1
        best_intersection_ratio = 0
        best_intersection_ratio_j = 0
        broken = False
        while j < count:
            bi_clust_j = bi_clust[j]
            bi_clust_j_size = len(bi_clust_j)
            bi_clust_intersection = bi_clust_i.intersection(bi_clust_j)
            intersection_size = len(bi_clust_intersection)
            if intersection_size == 0:
                j += 1
                continue
            if intersection_size == bi_clust_i_size:
                # Bi⊂Bj
                del bi_clust[i]
                count -= 1
                i -= 1
                broken = True
                break
            if intersection_size == bi_clust_j_size:
                # Bj⊂Bi
                del bi_clust[j]
                count -= 1
                continue
            average_intersection_ratio = (
                intersection_size
                * (bi_clust_j_size + bi_clust_i_size)
                / (bi_clust_j_size * bi_clust_i_size)
            )
            if average_intersection_ratio > best_intersection_ratio:
                best_intersection_ratio = average_intersection_ratio
                best_intersection_ratio_j = j
            j += 1

        if not broken and best_intersection_ratio > 0:
            if best_intersection_ratio >= merging_threshold:
                # Merge
                bi_clust[best_intersection_ratio_j] = bi_clust_i.union(
                    bi_clust[best_intersection_ratio_j]
                )
                del bi_clust[i]
                count -= 1
                continue
            # Split
            if bi_clust_i_size <= bi_clust_j_size:
                bi_clust[best_intersection_ratio_j] = (
                    bi_clust[best_intersection_ratio_j] - bi_clust_i
                )
                continue
            bi_clust[i] = bi_clust_i - bi_clust[best_intersection_ratio_j]
            continue
        i += 1


def _remove_subsets(bi_clust: list[set]) -> None:
    "Removes instances sets of bi_clust that are subsets or empty."

    i = 0
    count = len(bi_clust)
    while i < count - 1:
        if len(bi_clust[i]) == 0:
            del bi_clust[i]
            count -= 1
            continue
        bi_clust_i = bi_clust[i]
        j = i + 1
        while j < count:
            bi_clust_j = bi_clust[j]
            intersection_size = len(bi_clust_i.intersection(bi_clust_j))
            if intersection_size == len(bi_clust_i):
                # Bi⊂Bj
                del bi_clust[i]
                count -= 1
                i -= 1
                break
            if intersection_size == len(bi_clust_j):
                # Bj⊂Bi
                del bi_clust[j]
                count -= 1
                continue
            j += 1
        i += 1


def consensus_function_14(bi_clust: list[set], merging_threshold: float = 0.5):
    """Returns a modified bi_clust (set of unique instance sets)."""

    while True:
        _remove_subsets(bi_clust)
        bi_clust_size = len(bi_clust)
        if bi_clust_size == 1:
            return
        intersection_matrix = pd.DataFrame(
            columns=range(bi_clust_size), index=range(bi_clust_size), dtype=int
        )

        for i in range(bi_clust_size - 1):
            bi_clust_i = bi_clust[i]
            bi_clust_i_size = len(bi_clust_i)
            for j in range(i + 1, bi_clust_size):
                bi_clust_j = bi_clust[j]
                bi_clust_j_size = len(bi_clust_j)
                intersection_size = len(bi_clust_i.intersection(bi_clust_j))
                if intersection_size == 0:
                    continue
                intersection_matrix.iloc[i, j] = intersection_size / bi_clust_i_size
                intersection_matrix.iloc[j, i] = intersection_size / bi_clust_j_size

        if intersection_matrix.isna().values.all():
            break

        pointer = pd.DataFrame(columns=range(3), index=range(bi_clust_size))
        for i in range(bi_clust_size):
            if not intersection_matrix.iloc[i, :].isna().all():
                pointer.iloc[i, 0] = i
                pointer.iloc[i, 1] = intersection_matrix.iloc[i, :].argmax()
                pointer.iloc[i, 2] = intersection_matrix.iloc[i, pointer.iloc[i, 1]]

        pointer.sort_values(2, inplace=True, ascending=False)
        pointer = pointer[pointer.iloc[:, 2] > 0.0]

        for k in range(pointer.shape[0]):
            i = pointer.iloc[k, 0]
            j = pointer.iloc[k, 1]
            value = intersection_matrix.iloc[i, j]
            if value is None:
                continue
            if value >= merging_threshold:
                bi_clust[i] = bi_clust[i].union(bi_clust[j])
                bi_clust[j] = set()
                intersection_matrix.iloc[i, :] = None
                intersection_matrix.iloc[:, j] = None
                continue
            if len(bi_clust[i]) <= len(bi_clust[j]):
                bi_clust[j] = bi_clust[j] - bi_clust[i]
                intersection_matrix.iloc[j, :] = None
                intersection_matrix.iloc[:, j] = None
                continue
            bi_clust[i] = bi_clust[i] - bi_clust[j]
            intersection_matrix.iloc[i, :] = None
            intersection_matrix.iloc[:, i] = None


def consensus_function_15(bi_clust: list[set], merging_threshold: float = 0.5):
    """Returns a modified bi_clust (set of unique instance sets)."""

    _remove_subsets(bi_clust)
    bi_clust_size = len(bi_clust)
    if bi_clust_size == 1:
        return
    intersection_matrix = pd.DataFrame(
        0, columns=range(bi_clust_size), index=range(bi_clust_size), dtype=int
    )
    for i in range(bi_clust_size - 1):
        bi_clust_i = bi_clust[i]
        bi_clust_i_size = len(bi_clust_i)
        for j in range(i + 1, bi_clust_size):
            bi_clust_j = bi_clust[j]
            bi_clust_j_size = len(bi_clust_j)
            intersection_size = len(bi_clust_i.intersection(bi_clust_j))
            if intersection_size == 0:
                continue
            intersection_matrix.iloc[i, j] = intersection_size / bi_clust_i_size
            intersection_matrix.iloc[j, i] = intersection_size / bi_clust_j_size

    cluster_indexes = sp.coo_matrix(intersection_matrix >= merging_threshold).nonzero()
    for index, i in enumerate(cluster_indexes[0]):
        j = cluster_indexes[1][index]
        bi_clust[i] = bi_clust[j] = bi_clust[i].union(bi_clust[j])

    _remove_subsets(bi_clust)
    bi_clust_size = len(bi_clust)
    if bi_clust_size == 1:
        return

    for i in range(bi_clust_size - 1):
        bi_clust_i = bi_clust[i]
        bi_clust_i_size = len(bi_clust_i)
        for j in range(i + 1, bi_clust_size):
            bi_clust_j = bi_clust[j]
            bi_clust_j_size = len(bi_clust_j)
            bi_clust_intersection = bi_clust_i.intersection(bi_clust_j)
            if len(bi_clust_intersection) == 0:
                continue
            if bi_clust_i_size <= bi_clust_j_size:
                bi_clust[j] = bi_clust_j - bi_clust_intersection
                continue
            bi_clust[i] = bi_clust_i - bi_clust_intersection

    _remove_subsets(bi_clust)
