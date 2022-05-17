"""Core MultiCons definition"""

from typing import Callable, Literal, Union

import graphviz
import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.base import BaseEstimator

from .consensus import (
    consensus_function_10,
    consensus_function_12,
    consensus_function_13,
    consensus_function_14,
    consensus_function_15,
)
from .utils import (
    build_base_clusterings,
    build_bi_clust,
    build_membership_matrix,
    jaccard_index,
    jaccard_similarity,
    linear_closed_itemsets_miner,
)


class MultiCons(BaseEstimator):
    """MultiCons (Multiple Consensus) algorithm.

    MultiCons is a consensus clustering method that uses the frequent closed itemset
    mining technique to find similarities in the base clustering solutions.

    Args:
        consensus_function (str or function): Specifies a
            consensus function to generate clusters from the available instance sets at
            each iteration.
            Currently the following consensus functions are available:

            - `consensus_function_10`: The simplest approach, *used by default*.
                Removes instance sets with inclusion property and groups together
                intersecting instance sets.
            - `consensus_function_12`: Similar to `consensus_function_10`. Uses a
                `merging_threshold` to decide whether to merge the intersecting instance
                sets or to split them (removing the intersection from the
                bigger set).
            - `consensus_function_13`: A stricter version of `consensus_function_12`.
                Compares the maximal average intersection ratio with the
                `merging_threshold` to decide whether to merge the intersecting instance
                sets or to split them.
            - `consensus_function_14`: A version of `consensus_function_13` that first
                searches the maximal intersection ratio among all possible intersections
                prior applying a merge or split decision.
            - `consensus_function_15`: A graph based approach. Builds an adjacency
                matrix from the intersection matrix using the `decision_threshold`.
                Merges all connected nodes and then splits overlapping instance sets.

            To use another consensus function it is possible to pass a function instead
            of a string value. The function should accept two arguments - a list of sets
            and an optional `merging_threshold`, and should update the list of sets in
            place. See `consensus_function_10` for an example.
        similarity_measure (str or function): Specifies how to compute the similarity
            between two clustering solutions.
            Currently the following similarity measures are available:

            - `JaccardSimilarity`: Indicates that the pair-wise jaccard similarity
                measure should be used. This measure is computed with the formula
                `yy / (yy + ny + yn)`
                Where: `yy` is number of times two points belong to same cluster in both
                clusterings and `ny` and `yn` are the number of times two points belong
                to the same cluster in one clustering but not in the other.
            - `JaccardIndex`: Indicates that the set-wise jaccard similarity
                coefficient should be used. This measure is computed with the
                formula `|X ∩ Y| / (|X| + |Y| - |X ∩ Y|)`
                Where: X and Y are the clustering solutions.

            To use another similarity measure it is possible to pass a function instead
            of a string value. The function should accept two arguments - two numeric
            numpy arrays (representing the two clustering solutions) and should return a
            numeric score (indicating how similar the clustering solutions are).
        merging_threshold (float): Specifies the minimum required ratio (calculated from
            the intersection between two sets over the size of the smaller set) for
            which the `consensus_function` should merge two sets. Only applies to
            `consensus_function_12`.
        optimize_label_names (bool): Indicates whether the label assignment of
            the clustering partitions should be optimized to maximize the similarity
            measure score (using the Hungarian algorithm). By default set to `False` as
            the default `similarity_measure` score ("JaccardSimilarity") does not depend
            on which labels are assigned to which cluster.

    Attributes:
        consensus_function (function): The consensus function used to generate clusters
            from the available instance sets at each iteration.
        consensus_vectors (list of numpy arrays): The list of proposed consensus
            clustering candidates.
        decision_thresholds (list of int): The list of decision thresholds values,
            corresponding to the consensus vectors (in the same order). A decision
            threshold indicates how many base clustering solutions were required to
            agree (at least) to form sub-clusters.
        ensemble_similarity (list of float): The list of ensemble similarity measures
            corresponding to the consensus vectors.
        labels_ (numpy array): The recommended consensus candidate.
        optimize_label_names (bool): Indicates whether the label assignment of
            clustering partitions should be optimized or not.
        recommended (int): The index of the recommended consensus vector.
        similarity_measure (function): The similarity function used to measure the
            similarity between two clustering solutions.
        stability (list of int): The list of stability values, corresponding to the
            consensus vectors (in the same order). A stability value indicates how many
            times the same consensus is generated for different decision thresholds.
        tree_quality (float): The tree quality measure (between 0 and 1). Higher is
            better.

    Raises:
        ValueError: If `consensus_function` or `similarity_measure` is not a function
            and not one of the allowed string values (mentioned above).
    """

    # pylint: disable=too-many-instance-attributes

    _consensus_functions = {
        "consensus_function_10": consensus_function_10,
        "consensus_function_12": consensus_function_12,
        "consensus_function_13": consensus_function_13,
        "consensus_function_14": consensus_function_14,
        "consensus_function_15": consensus_function_15,
    }
    _similarity_measures = {
        "JaccardSimilarity": jaccard_similarity,
        "JaccardIndex": jaccard_index,
    }

    def __init__(
        self,
        consensus_function: Union[
            Literal[
                "consensus_function_10",
                "consensus_function_12",
                "consensus_function_13",
                "consensus_function_14",
                "consensus_function_15",
            ],
            Callable[[list[np.ndarray]], None],
        ] = "consensus_function_10",
        merging_threshold: float = 0.5,
        similarity_measure: Union[
            Literal["JaccardSimilarity", "JaccardIndex"],
            Callable[[np.ndarray, np.ndarray], int],
        ] = "JaccardSimilarity",
        optimize_label_names: bool = False,
    ):
        """Initializes MultiCons."""

        self.consensus_function = self._parse_argument(
            "consensus_function", self._consensus_functions, consensus_function
        )
        self.similarity_measure = self._parse_argument(
            "similarity_measure", self._similarity_measures, similarity_measure
        )
        self.merging_threshold = merging_threshold
        self.optimize_label_names = optimize_label_names
        self.consensus_vectors = None
        self.decision_thresholds = None
        self.ensemble_similarity = None
        self.labels_ = None
        self.recommended = None
        self.stability = None
        self.tree_quality = None

    def fit(self, X, y=None, sample_weight=None):  # pylint: disable=unused-argument
        """Computes the MultiCons consensus.

        Args:
            X (list of numeric numpy arrays or a pandas Dataframe): Either a list of
                arrays where each array represents one clustering solution
                (base clusterings), or a Dataframe representing a binary membership
                matrix.
            y: Ignored. Not used, present here for API consistency by convention.
            sample_weight: Ignored. Not used, present here for API consistency by
                convention.

        Returns:
            self: Returns the (fitted) instance itself.
        """

        if isinstance(X, pd.DataFrame):
            membership_matrix = pd.DataFrame(X, dtype=bool)
            X = build_base_clusterings(X)
        else:
            X = np.array(X, dtype=int)
            # 2 Calculate in-ensemble similarity
            # similarity = in_ensemble_similarity(X)
            # 3 Build the cluster membership matrix M
            membership_matrix = build_membership_matrix(X)

        # 4 Generate FCPs from M for minsupport = 0
        # 5 Sort the FCPs in ascending order according to the size of the instance sets
        frequent_closed_itemsets = linear_closed_itemsets_miner(membership_matrix)
        # 6 MaxDT ← length(BaseClusterings)
        max_d_t = len(X)
        # 7 BiClust ← {instance sets of FCPs built from MaxDT base clusters}
        bi_clust = build_bi_clust(membership_matrix, frequent_closed_itemsets, max_d_t)
        # 8 Assign a label to each set in BiClust to build the first consensus vector
        #   and store it in a list of vectors ConsVctrs
        self.consensus_vectors = [0] * max_d_t
        self.consensus_vectors[max_d_t - 1] = self._assign_labels(bi_clust, X)

        # 9 Build the remaining consensuses
        # 10 for DT = (MaxDT−1) to 1 do
        for d_t in range(max_d_t - 1, 0, -1):
            # 11 BiClust ← BiClust ∪ {instance sets of FCPs built from DT base clusters}
            bi_clust += build_bi_clust(membership_matrix, frequent_closed_itemsets, d_t)
            # 12 Call the consensus function (Algo. 10)
            self.consensus_function(bi_clust, self.merging_threshold)
            # 13 Assign a label to each set in BiClust to build a consensus vector
            #    and add it to ConsVctrs
            self.consensus_vectors[d_t - 1] = self._assign_labels(bi_clust, X)
        # 14 end

        # 15 Remove similar consensuses
        # 16 ST ← Vector of ‘1’s of length MaxDT
        self.decision_thresholds = list(range(1, max_d_t + 1))
        self.stability = [1] * max_d_t
        # 17 for i = MaxDT to 2 do
        i = max_d_t - 1
        while i > 0:
            # 18 Vi ← ith consensus in ConsVctrs
            consensus_i = self.consensus_vectors[i]
            # 19 for j = (i−1) to 1 do
            j = i - 1
            while j >= 0:
                # 20 Vj ← jth consensus in ConsVctrs
                consensus_j = self.consensus_vectors[j]
                # 21 if Jaccard(Vi , Vj ) = 1 then
                if self.similarity_measure(consensus_i, consensus_j) == 1:
                    # 22 ST [i] ← ST [i] + 1
                    self.stability[i] += 1
                    # 23 Remove ST [j]
                    del self.stability[j]
                    del self.decision_thresholds[j]
                    # 24 Remove Vj from ConsVctrs
                    del self.consensus_vectors[j]
                    i -= 1
                j -= 1
            i -= 1
            # 25 end
        # 26 end

        # 27 Find the consensus the most similar to the ensemble
        # 28 L ← length(ConsVctrs)
        consensus_count = len(self.consensus_vectors)
        # 29 TSim ← Vector of ‘0’s of length L
        t_sim = np.zeros(consensus_count)
        # 30 for i = 1 to L do
        for i in range(consensus_count):
            # 31 Ci ← ith consensus in ConsVctrs
            consensus_i = self.consensus_vectors[i]
            # 32 for j = 1 to MaxDT do
            for j in range(max_d_t):
                # 33 Cj ← jth clustering in BaseClusterings
                consensus_j = X[j]
                # 34 TSim[i] ← TSim[i] + Jaccard(Ci,Cj)
                t_sim[i] += self.similarity_measure(consensus_i, consensus_j)
            # 35 end
            # 36 Sim[i] ← TSim[i] / MaxDT
            t_sim[i] /= max_d_t
        # 37 end
        self.recommended = np.where(t_sim == np.amax(t_sim))[0][0]
        self.labels_ = self.consensus_vectors[self.recommended]

        self.tree_quality = 1
        if len(np.unique(self.consensus_vectors[0])) == 1:
            self.tree_quality -= (self.stability[0] - 1) / max(self.decision_thresholds)

        self.ensemble_similarity = t_sim
        return self

    def cons_tree(self) -> graphviz.Digraph:
        """Returns a ConsTree graph. Requires the `fit` method to be called first."""

        graph = graphviz.Digraph()
        graph.attr(
            "graph", label=f"ConsTree\nTree Quality = {self.tree_quality}", labelloc="t"
        )
        unique_count = [
            np.unique(vec, return_counts=True) for vec in self.consensus_vectors
        ]
        max_size = len(self.consensus_vectors[0])

        previous = []
        for i, nodes_count in enumerate(unique_count):
            attributes = {
                "fillcolor": "slategray2",
                "shape": "ellipse",
                "style": "filled",
            }
            if i == self.recommended:
                attributes.update({"fillcolor": "darkseagreen", "shape": "box"})
            for j in range(len(nodes_count[0])):
                node_id = f"{i}{nodes_count[0][j]}"
                attributes["width"] = str(int(9 * nodes_count[1][j] / max_size))
                graph.attr("node", **attributes)
                graph.node(node_id, str(nodes_count[1][j]))
                if i == 0:
                    continue
                for node in np.unique(
                    previous[self.consensus_vectors[i] == nodes_count[0][j]]
                ):
                    graph.edge(f"{i - 1}{node}", node_id)

            previous = self.consensus_vectors[i]
            with graph.subgraph(name="cluster") as sub_graph:
                sub_graph.attr("graph", label="Legend")
                sub_graph.attr("node", shape="box", width="")
                values = [
                    f"DT={self.decision_thresholds[i]}",
                    f"ST={self.stability[i]}",
                    f"Similarity={round(self.ensemble_similarity[i], 2)}",
                ]
                sub_graph.node(f"legend_{i}", " ".join(values))
                if i > 0:
                    sub_graph.edge(f"legend_{i-1}", f"legend_{i}")
        return graph

    def _assign_labels(self, bi_clust: list[set], base_clusterings: np.ndarray):
        """Returns a consensus vector with labels for each instance set in bi_clust."""

        result = np.zeros(len(base_clusterings[0]), dtype=int)
        if not self.optimize_label_names:
            for i, itemset in enumerate(bi_clust):
                result[list(itemset)] = i
            return result
        unique_labels = np.unique(base_clusterings.flatten()).tolist()
        max_label = max(unique_labels)
        for i in range(len(bi_clust) - len(unique_labels)):
            unique_labels.append(max_label + i + 1)

        cost_matrix = pd.DataFrame(
            0.0, index=range(len(bi_clust)), columns=unique_labels
        )
        for i, itemset in enumerate(map(list, bi_clust)):
            for j, label in enumerate(unique_labels):
                labels = np.ones(len(itemset)) * label
                score = np.array(
                    [
                        self.similarity_measure(clustering[itemset], labels)
                        for clustering in base_clusterings
                    ]
                )
                cost_matrix.loc[i, j] = len(itemset) * (1 + score).sum()

        col_ind = linear_sum_assignment(
            cost_matrix.apply(lambda x: x.max() - x, axis=1)
        )[1]

        for i, itemset in enumerate(bi_clust):
            result[list(itemset)] = col_ind[i]
        return result

    @staticmethod
    def _parse_argument(name, arguments, argument) -> Callable:
        """Returns the function that corresponds to the argument."""

        if callable(argument):
            return argument
        value = arguments.get(argument, None)
        if not value:
            raise ValueError(
                f"Invalid value for `{name}` argument. "
                f"Should be one of ({', '.join(arguments.keys())}) or a function. "
                f"But received `{argument}` instead."
            )
        return value
