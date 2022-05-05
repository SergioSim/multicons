"""Utility functions"""

from typing import Callable

import graphviz
import numpy as np
import pandas as pd
from fim import eclat  # pylint: disable=no-name-in-module
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import jaccard_score


def jaccard_index(prediction: np.ndarray, true_labels: np.ndarray):
    """Returns the jaccard index using the formula: |X ∩ Y| / (|X| + |Y| - |X ∩ Y|)."""

    intersection_size = (prediction == true_labels).sum()
    return intersection_size / (len(true_labels) + len(prediction) - intersection_size)


def sklearn_jaccard_score(prediction: np.ndarray, true_labels: np.ndarray):
    """Wraps the scikit-learn `jaccard_score` function with weighted average."""

    return jaccard_score(prediction, true_labels, average="weighted")


def ensemble_jaccard_score(prediction: np.ndarray, true_labels: np.ndarray):
    """Computes a jaccard score without considering the underlining labels."""

    prediction_sets = [set(np.where(prediction == x)[0]) for x in np.unique(prediction)]
    labels_sets = [set(np.where(true_labels == x)[0]) for x in np.unique(true_labels)]
    score = 0
    for itemset in prediction_sets:
        intersections = []
        for itemset_j in labels_sets:
            intersections.append(
                len(itemset.intersection(itemset_j)) / len(itemset.union(itemset_j))
            )
        score += max(intersections) - min(intersections)
    return 2 * score / (len(prediction_sets) + len(labels_sets))


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
    index = np.arange(count)
    similarity = pd.DataFrame(0.0, index=index, columns=index)
    average_similarity = np.zeros(count)
    for i in range(count - 1):
        cluster_i = base_clusterings[i]
        for j in range(i + 1, count):
            cluster_j = base_clusterings[j]
            score = sklearn_jaccard_score(cluster_i, cluster_j)
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


def assign_labels(
    bi_clust: list[set],
    base_clusterings: np.ndarray,
    similarity_measure: Callable[
        [np.ndarray, np.ndarray], int
    ] = ensemble_jaccard_score,
):
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
                    similarity_measure(clustering[itemset], labels)
                    for clustering in base_clusterings
                ]
            )
            cost_matrix.loc[i, j] = itemset_len * (1 + score).sum()

    col_ind = linear_sum_assignment(cost_matrix.apply(lambda x: x.max() - x, axis=1))[1]

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


def multicons(
    base_clusterings: list[np.ndarray],
    consensus_function: Callable[[list[np.ndarray]], None] = consensus_function_10,
    similarity_measure: Callable[
        [np.ndarray, np.ndarray], int
    ] = ensemble_jaccard_score,
):
    """Returns a dictionary with a list of consensus clustering vectors."""
    # pylint: disable=too-many-locals

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
    consensus_vectors[max_d_t - 1] = assign_labels(
        bi_clust, base_clusterings, similarity_measure
    )

    # 9 Build the remaining consensuses
    # 10 for DT = (MaxDT−1) to 1 do
    for d_t in range(max_d_t - 1, 0, -1):
        # 11 BiClust ← BiClust ∪ {instance sets of FCPs built from DT base clusters}
        bi_clust += build_bi_clust(membership_matrix, frequent_closed_itemsets, d_t)
        # 12 Call the consensus function (Algo. 10)
        consensus_function(bi_clust)
        # 13 Assign a label to each set in BiClust to build a consensus vector
        #    and add it to ConsVctrs
        consensus_vectors[d_t - 1] = assign_labels(
            bi_clust, base_clusterings, similarity_measure
        )
    # 14 end

    # 15 Remove similar consensuses
    # 16 ST ← Vector of ‘1’s of length MaxDT
    decision_thresholds = list(range(1, max_d_t + 1))
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
            if similarity_measure(consensus_i, consensus_j) == 1:
                # 22 ST [i] ← ST [i] + 1
                stability[i] += 1
                # 23 Remove ST [j]
                del stability[j]
                del decision_thresholds[j]
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
    for i in range(consensus_count):
        # 31 Ci ← ith consensus in ConsVctrs
        consensus_i = consensus_vectors[i]
        # 32 for j = 1 to MaxDT do
        for j in range(max_d_t):
            # 33 Cj ← jth clustering in BaseClusterings
            consensus_j = base_clusterings[j]
            # 34 TSim[i] ← TSim[i] + Jaccard(Ci,Cj)
            t_sim[i] += similarity_measure(consensus_i, consensus_j)
        # 35 end
        # 36 Sim[i] ← TSim[i] / MaxDT
        t_sim[i] /= max_d_t
    # 37 end
    recommended = np.where(t_sim == np.amax(t_sim))[0][0]

    tree_quality = 1
    if len(np.unique(consensus_vectors[0])) == 1:
        tree_quality = 1 - (stability[0] - 1) / max(decision_thresholds)

    return {
        "consensus_vectors": consensus_vectors,
        "decision_thresholds": decision_thresholds,
        "recommended": recommended,
        "stability": stability,
        "tree_quality": tree_quality,
        "ensemble_similarity": t_sim,
    }


def cons_tree(consensus: dict):
    """Receives the result of multicons `consensus` and returns a ConsTree graph."""

    graph = graphviz.Digraph()
    tree_quality = consensus["tree_quality"]
    graph.attr("graph", label=f"ConsTree\nTree Quality = {tree_quality}", labelloc="t")

    consensus_vectors = consensus["consensus_vectors"]
    unique_count = [np.unique(vec, return_counts=True) for vec in consensus_vectors]
    max_size = len(consensus_vectors[0])

    previous = []
    for i, nodes_count in enumerate(unique_count):
        attributes = {"fillcolor": "slategray2", "shape": "ellipse", "style": "filled"}
        if i == consensus["recommended"]:
            attributes.update({"fillcolor": "darkseagreen", "shape": "box"})
        for j in range(len(nodes_count[0])):
            node_id = f"{i}{nodes_count[0][j]}"
            attributes["width"] = str(int(9 * nodes_count[1][j] / max_size))
            graph.attr("node", **attributes)
            graph.node(node_id, str(nodes_count[1][j]))
            if i == 0:
                continue
            for node in np.unique(previous[consensus_vectors[i] == nodes_count[0][j]]):
                graph.edge(f"{i - 1}{node}", node_id)

        previous = consensus_vectors[i]
        with graph.subgraph(name="cluster") as sub_graph:
            sub_graph.attr("graph", label="Legend")
            sub_graph.attr("node", shape="box", width="")
            values = [
                f"DT={consensus['decision_thresholds'][i]}",
                f"ST={consensus['stability'][i]}",
                f"Similarity={round(consensus['ensemble_similarity'][i], 2)}",
            ]
            sub_graph.node(f"legend_{i}", " ".join(values))
            if i > 0:
                sub_graph.edge(f"legend_{i-1}", f"legend_{i}")
    return graph
