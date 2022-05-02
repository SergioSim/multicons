"""Utility functions"""

import numpy as np
import pandas as pd
from fim import eclat  # pylint: disable=no-name-in-module
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import jaccard_score


def build_membership_matrix(base_clusterings: np.ndarray) -> pd.DataFrame:
    """Computes and returns the membership matrix."""

    if len(base_clusterings) == 0 or not isinstance(base_clusterings[0], np.ndarray):
        raise IndexError("base_clusterings should contain at least one np.ndarray.")

    res = []
    for clusters in base_clusterings:
        res += [clusters == x for x in np.unique(clusters)]
    return pd.DataFrame(np.transpose(res), dtype=int)


def in_ensemble_similarity(base_clusterings: list[np.ndarray]) -> float:
    """Returns the average similarity among the base clusters using Jaccard score."""

    if not base_clusterings or len(base_clusterings) < 2:
        raise IndexError("base_clusterings should contain at least two np.ndarrays.")

    count = len(base_clusterings)
    index = np.arange(0, count)
    similarity = pd.DataFrame(0.0, index=index, columns=index)
    average_similarity = np.zeros(count)
    for i in range(0, count - 1):
        cluster_i = base_clusterings[i]
        for j in range(i + 1, count):
            cluster_j = base_clusterings[j]
            score = jaccard_score(cluster_i, cluster_j, average="weighted")
            similarity.iloc[i, j] = similarity.iloc[j, i] = score
        average_similarity[i] = similarity.iloc[i].sum() / (count - 1)

    average_similarity[count - 1] = similarity.iloc[count - 1].sum() / (count - 1)
    return np.mean(average_similarity)


def linear_closed_itemsets_miner(membership_matrix: pd.DataFrame):
    """Returns a list of frequent closed itemsets using the lcm algorithm."""

    transactions = []
    for i in membership_matrix.index:
        transactions.append(np.nonzero(membership_matrix.iloc[i].values)[0].tolist())
    frequent_closed_itemsets = eclat(transactions, target="c", supp=0, algo="o", conf=0)
    return sorted(map(lambda x: frozenset(x[0]), frequent_closed_itemsets), key=len)


def assign_labels(bi_clust: list[set], base_clusterings: np.ndarray):
    """Returns a consensus vector with labels for each instance set in bi_clust."""

    unique_labels = np.unique(base_clusterings.flatten())
    max_label = unique_labels.max()
    unique_labels = unique_labels.tolist()
    for i in range(len(bi_clust) - len(unique_labels)):
        unique_labels.append(max_label + i + 1)

    cost_matrix = pd.DataFrame(0.0, index=range(len(bi_clust)), columns=unique_labels)
    for i, itemset in enumerate(map(list, bi_clust)):
        itemset_len = len(itemset)
        for j, label in enumerate(unique_labels):
            labels = np.ones(itemset_len) * label
            score = np.array(
                [
                    jaccard_score(clustering[itemset], labels, average="weighted")
                    for clustering in base_clusterings
                ]
            )
            cost_matrix.loc[i, j] = itemset_len * (1 + score).sum()

    _, col_ind = linear_sum_assignment(cost_matrix.apply(lambda x: x.max() - x, axis=1))

    result = np.zeros(len(base_clusterings[0]), dtype=int)
    for i, itemset in enumerate(bi_clust):
        result[list(itemset)] = col_ind[i]
    return result


def consensus_function_10(bi_clust: list[np.ndarray]):
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


def build_bi_clust(
    membership_matrix: pd.DataFrame,
    frequent_closed_itemsets: list[frozenset],
    size: int,
):
    """Returns a new list of instances from itemsets of given size."""

    itemsets = list(filter(lambda x: len(x) == size, frequent_closed_itemsets))
    result = []
    for itemset in itemsets:
        consensus = np.ones(len(membership_matrix), dtype=bool)
        for partition in itemset:
            consensus = consensus & membership_matrix[partition]
        result.append(set(consensus[consensus].index.values))
    return result


def multicons(base_clusterings: list[np.ndarray]):
    """Returns a dictionary with a list of consensus clustering vectors."""

    base_clusterings = np.array(base_clusterings)
    # 2 Calculate in-ensemble similarity
    # similarity = in_ensemble_similarity(base_clusterings)
    # 3 Build the cluster membership matrix M
    membership_matrix = build_membership_matrix(base_clusterings)
    # 4 Generate FCPs from M for minsupport = 0
    # 5 Sort the FCPs in ascending order according to the size of the instance sets
    frequent_closed_itemsets = linear_closed_itemsets_miner(membership_matrix)
    # 6 MaxDT ← length(BaseClusterings)
    max_d_t = len(base_clusterings)
    # 7 BiClust ← {instance sets of FCPs built from MaxDT base clusters}
    bi_clust = build_bi_clust(membership_matrix, frequent_closed_itemsets, max_d_t)
    # 8 Assign a label to each set in BiClust to build the first consensus vector
    #   and store it in a list of vectors ConsVctrs
    consensus_vectors = [0] * max_d_t
    consensus_vectors[max_d_t - 1] = assign_labels(bi_clust, base_clusterings)

    # 9 Build the remaining consensuses
    # 10 for DT = (MaxDT−1) to 1 do
    for d_t in range(max_d_t - 1, 0, -1):
        # 11 BiClust ← BiClust ∪ {instance sets of FCPs built from DT base clusters}
        bi_clust += build_bi_clust(membership_matrix, frequent_closed_itemsets, d_t)
        # 12 Call the consensus function (Algo. 10)
        consensus_function_10(bi_clust)
        # 13 Assign a label to each set in BiClust to build a consensus vector
        #    and add it to ConsVctrs
        consensus_vectors[d_t - 1] = assign_labels(bi_clust, base_clusterings)
    # 14 end

    # 15 Remove similar consensuses
    # 16 ST ← Vector of ‘1’s of length MaxDT
    stability = [1] * max_d_t
    # 17 for i = MaxDT to 2 do
    i = max_d_t - 1
    while i > 0:
        # 18 Vi ← ith consensus in ConsVctrs
        consensus_i = consensus_vectors[i]
        # 19 for j = (i−1) to 1 do
        j = i - 1
        while j >= 0:
            # 20 Vj ← jth consensus in ConsVctrs
            consensus_j = consensus_vectors[j]
            # 21 if Jaccard(Vi , Vj ) = 1 then
            if jaccard_score(consensus_i, consensus_j, average="weighted") == 1:
                # 22 ST [i] ← ST [i] + 1
                stability[i] += 1
                # 23 Remove ST [j]
                del stability[j]
                # 24 Remove Vj from ConsVctrs
                del consensus_vectors[j]
                i -= 1
            j -= 1
        i -= 1
        # 25 end
    # 26 end

    # 27 Find the consensus the most similar to the ensemble
    # 28 L ← length(ConsVctrs)
    consensus_count = len(consensus_vectors)
    # 29 TSim ← Vector of ‘0’s of length L
    t_sim = np.zeros(consensus_count)
    # 30 for i = 1 to L do
    for i in range(0, consensus_count):
        # 31 Ci ← ith consensus in ConsVctrs
        consensus_i = consensus_vectors[i]
        # 32 for j = 1 to MaxDT do
        for j in range(max_d_t):
            # 33 Cj ← jth clustering in BaseClusterings
            consensus_j = base_clusterings[j]
            # 34 TSim[i] ← TSim[i] + Jaccard(Ci,Cj)
            t_sim[i] += jaccard_score(consensus_i, consensus_j, average="weighted")
        # 35 end
        # 36 Sim[i] ← TSim[i] / MaxDT
        t_sim[i] /= max_d_t
    # 37 end
    recommended = np.where(t_sim == np.amax(t_sim))[0][0]
    return {
        "recommended": recommended,
        "consensus_vectors": consensus_vectors,
        "t_sim": t_sim,
    }
