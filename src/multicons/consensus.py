"""Consensus functions definitions."""

# pylint: disable=unused-argument


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
