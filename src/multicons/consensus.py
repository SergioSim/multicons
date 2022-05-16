"""Consensus functions definitions."""


def consensus_function_10(bi_clust: list[set]):
    """Returns a modified bi_clust (set of unique instance sets)."""

    i = 0
    count = len(bi_clust)
    while i < count:
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
