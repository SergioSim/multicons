"""Consensus functions definitions."""


def consensus_function_10(bi_clust: list[set]):
    """Returns a modified bi_clust (set of unique instance sets)."""

    all_bi_clust_sets_unique = False
    while not all_bi_clust_sets_unique:
        all_bi_clust_sets_unique = True
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
                elif intersection_size == len(bi_clust_i):
                    # Bi⊂Bj
                    all_bi_clust_sets_unique = False
                    del bi_clust[i]
                    count -= 1
                elif intersection_size == len(bi_clust_j):
                    # Bj⊂Bi
                    all_bi_clust_sets_unique = False
                    del bi_clust[j]
                    count -= 1
                else:
                    bi_clust[j] = bi_clust_i.union(bi_clust_j)
                    all_bi_clust_sets_unique = False
                    del bi_clust[i]
                    count -= 1
            i += 1
