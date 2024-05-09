# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.2
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# # Examples
#
# In this section we present some usage examples for MultiCons.
# We replicate the examples presented in the Thesis of Atheer Al-Najdi
# (A closed patterns-based approach to the consensus clustering problem).

# %% [markdown]
# ## Cassini dataset
#
# The dataset consists of 1000 instances, each represents a point in a 2D space,
# forming a structure of three clusters.
#
# As in the Thesis of Atheer Al-Najdi, we will try to cluster the dataset with 8
# different clustering algorithms and then compute and visualize the consensus
# clustering candidates using the MultiCons and ConsTree methods.
#
# **Let's get a first visual of the dataset and our base clusterings:**

# %%
from os import path

import numpy as np
import pandas as pd
from fcmeans import FCM
from kmedoids import KMedoids
from matplotlib import pyplot as plt
from sklearn.cluster import (
    DBSCAN,
    AgglomerativeClustering,
    Birch,
    KMeans,
    SpectralClustering,
)
from sklearn.mixture import GaussianMixture

from multicons import MultiCons
from multicons.utils import jaccard_similarity

np.set_printoptions(threshold=100)

# %%
# Load the data
file_prefix = "" if path.exists("cassini.csv") else "docs/"
file_name = f"{file_prefix}cassini.csv"
cassini = pd.read_csv(file_name)
# Remove the class labels
cassini_train_data = cassini.drop(['class'], axis=1)

# %%
# Setup the plot axes
fig, axes = plt.subplots(
    nrows=3, ncols=3, figsize=(18, 12), sharex=True, sharey=True
)
# Common plot arguments
common_kwargs = {"x": "x", "y": "y", "colorbar": False, "colormap": "Paired"}
# Our collection of base clusterings
base_clusterings = []

# Cassini
cassini.plot.scatter(c="class", title="Cassini", ax=axes[0, 0], **common_kwargs)

# K-means
base_clusterings.append(KMeans(n_clusters=3).fit_predict(cassini_train_data))
cassini_train_data.plot.scatter(
    title="K-means", ax=axes[0, 1], c=base_clusterings[-1], **common_kwargs
)

# Average linkage
base_clusterings.append(
    AgglomerativeClustering(n_clusters=3).fit_predict(cassini_train_data)
)
cassini_train_data.plot.scatter(
    title="Average linkage", ax=axes[0, 2], c=base_clusterings[-1], **common_kwargs
)

# Gaussian model
base_clusterings.append(
    GaussianMixture(n_components=3, random_state=5).fit_predict(cassini_train_data)
)
cassini_train_data.plot.scatter(
    title="Gaussian model", ax=axes[1, 0], c=base_clusterings[-1], **common_kwargs
)

# C-means
fcm = FCM(n_clusters=3, max_iter=5, m=5)
fcm.fit(cassini_train_data.values)
base_clusterings.append(fcm.predict(cassini_train_data.values))
cassini_train_data.plot.scatter(
    title="C-means", ax=axes[1, 1], c=base_clusterings[-1], **common_kwargs
)

# PAM
base_clusterings.append(
    KMedoids(3, metric="euclidean", method="pam")
    .fit_predict(cassini_train_data.to_numpy())
)
cassini_train_data.plot.scatter(
    title="PAM", ax=axes[1, 2], c=base_clusterings[-1], **common_kwargs
)

# BIRCH
birch = Birch(n_clusters=3, threshold=0.5)
base_clusterings.append(birch.fit_predict(np.ascontiguousarray(cassini_train_data)))
cassini_train_data.plot.scatter(
    title="BIRCH", ax=axes[2, 0], c=base_clusterings[-1], **common_kwargs
)

# Spectral
base_clusterings.append(
    SpectralClustering(n_clusters=3).fit_predict(cassini_train_data)
)
cassini_train_data.plot.scatter(
    title="Spectral", ax=axes[2, 1], c=base_clusterings[-1], **common_kwargs
)

# DBSCAN
base_clusterings.append(DBSCAN(eps=0.2).fit_predict(cassini_train_data))
cassini_train_data.plot.scatter(
    title="DBSCAN", ax=axes[2, 2], c=base_clusterings[-1], **common_kwargs
)

fig.show()

# %% [markdown]
# At this point, `base_clusterings` now contains all clustering candidates in a list
# (of lists):

# %%
np.array(base_clusterings)

# %% [markdown]
# > Note: This MultiCons implementation **requires** the clustering labels to be
# > **numerical**!

# %% [markdown]
# **Now, let's compute the consensus candidates with MultiCons:**

# %%
# MultiCons implementation aims to follow scikit-learn conventions.
consensus = MultiCons().fit(base_clusterings)
consensus

# %%
# The `consensus_vectors` attribute is a python list containing the
# consensus candidates.
# We transform it to a numpy array to better visualize it here.
np.array(consensus.consensus_vectors)

# %%
# The `decision_thresholds` attribute contains a list of decision thresholds
# for each consensus vector.
consensus.decision_thresholds

# %%
# The `recommended` attribute contains the index of the recommended consensus
# vector
consensus.recommended

# %%
# The `labels_` attribute contains the recommended consensus vector
consensus.labels_

# %%
# Plot the recommended consensus clustering solution
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
cassini_train_data.plot.scatter(
    title="MultiCons", ax=axes, c=consensus.labels_, **common_kwargs
)
fig.show()

# %%
# The `stability` attribute contains a list of stability values
# for each consensus vector.
consensus.stability

# %%
# The `tree_quality` member contains a measure of the tree quality.
# The measure ranges between 0 and 1. Higher is better.
consensus.tree_quality

# %%
# The `ensemble_similarity` contains a list of ensemble similarity measures
# for each consensus vector.
# They are between 0 and 1. Higher is better.
consensus.ensemble_similarity

# %% [markdown]
# **Finally, let's visualize the consenus candidates using the ConsTree method:**

# %%
cons_tree = consensus.cons_tree()
cons_tree

# %%
# Save the ConsTree graph to a file
cons_tree.render(outfile=f"{file_prefix}CassiniConsTree.svg", cleanup=True)

# %% [markdown]
# View ConsTree graph in full size: [CassiniConsTree.svg](../CassiniConsTree.svg)

# %% [markdown]
# ## 5 Overlapping Gaussian distributions
#
# Replicating the example with a synthetic dataset used in the thesis of Atheer
# that consist of:
# - generating 5 overlapping Gaussian distributed points in a 2D features space
# - appying 6 different clustering algorithms with random choices for K values
#     (in the range \[2, 9\])
# - comparing the results of 5 different MultiCons consensus solutions (by
#     altering the consensus functions)
#
# **Let's start by generating the dataset:**

# %%
cov = np.array([[0.3, 0.1], [0.1, 0.3]])
gaussian_distributions = pd.DataFrame(
    np.concatenate(
        (
            np.concatenate(
                (
                    np.random.multivariate_normal([-0.25, -2], cov, 400),
                    np.random.multivariate_normal([-1.75, -0.5], cov, 400),
                    np.random.multivariate_normal([-0.5, 2], cov, 400),
                    np.random.multivariate_normal([1.75, 2], cov, 400),
                    np.random.multivariate_normal([2, -0.5], cov, 400),
                ),
            ),
            np.concatenate(
                (
                    np.repeat([[1]], 400, 0),
                    np.repeat([[2]], 400, 0),
                    np.repeat([[3]], 400, 0),
                    np.repeat([[4]], 400, 0),
                    np.repeat([[5]], 400, 0),
                )
            )
        ),
        axis=1
    ),
    columns=["x", "y", "class"],
)
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(6, 4))
gaussian_distributions.plot.scatter(
    title="5 Overlapping Gaussians", ax=axes, c="class", **common_kwargs
)
fig.show()

# %%
# Remove the class labels
gaussian_train_data = gaussian_distributions.drop(['class'], axis=1)

# %% [markdown]
# **Next, let's compute the base clusterings and visualize their outcome:**

# %%
# Setup the plot axes
fig, axes = plt.subplots(
    nrows=3, ncols=2, figsize=(12, 12), sharex=True, sharey=True
)
# Our collection of base clusterings
base_clusterings = []

# K-means (4 clusters)
base_clusterings.append(KMeans(n_clusters=4).fit_predict(gaussian_train_data))
gaussian_train_data.plot.scatter(
    title="K-means (4 clusters)",
    ax=axes[0, 0],
    c=base_clusterings[-1],
    **common_kwargs
)

# Average linkage (9 clusters)
base_clusterings.append(
    AgglomerativeClustering(n_clusters=9, linkage="average").fit_predict(
        gaussian_train_data
    )
)
gaussian_train_data.plot.scatter(
    title="Average linkage (9 clusters)",
    ax=axes[0, 1],
    c=base_clusterings[-1],
    **common_kwargs
)

# Gaussian model (8 clusters)
base_clusterings.append(
    GaussianMixture(n_components=8, random_state=2, reg_covar=0.2).fit_predict(
        gaussian_train_data
    )
)
gaussian_train_data.plot.scatter(
    title="Gaussian model (8 clusters)",
    ax=axes[1, 0],
    c=base_clusterings[-1],
    **common_kwargs
)

# C-means (2 clusters)
fcm = FCM(n_clusters=2, max_iter=5, m=5)
fcm.fit(gaussian_train_data.values)
base_clusterings.append(fcm.predict(gaussian_train_data.values))
gaussian_train_data.plot.scatter(
    title="C-means (2 clusters)",
    ax=axes[1, 1],
    c=base_clusterings[-1],
    **common_kwargs
)

# PAM (3 clusters)
base_clusterings.append(
    KMedoids(3, metric="euclidean", method="pam")
    .fit_predict(gaussian_train_data.to_numpy())
)
gaussian_train_data.plot.scatter(
    title="PAM (3 clusters)", ax=axes[2, 0], c=base_clusterings[-1], **common_kwargs
)

# BIRCH (5 clusters)
birch = Birch(n_clusters=6, threshold=0.5)
base_clusterings.append(birch.fit_predict(gaussian_train_data))
gaussian_train_data.plot.scatter(
    title="BIRCH (6 clusters)",
    ax=axes[2, 1],
    c=base_clusterings[-1],
    **common_kwargs
)

fig.show()

# %% [markdown]
# **Now, let's compute the consensus candidates with MultiCons and visualize their
# outcome:**

# %%
consensus_1 = MultiCons()
consensus_2 = MultiCons(consensus_function="consensus_function_12")
consensus_3 = MultiCons(consensus_function="consensus_function_13")
consensus_4 = MultiCons(consensus_function="consensus_function_14")
consensus_5 = MultiCons(consensus_function="consensus_function_15")

consensus_1.fit(base_clusterings)
consensus_2.fit(base_clusterings)
consensus_3.fit(base_clusterings)
consensus_4.fit(base_clusterings)
consensus_5.fit(base_clusterings)

# Plot the recommended consensus clustering solutions
fig, axes = plt.subplots(nrows=3, ncols=2, figsize=(12, 12), sharex=True, sharey=True)
gaussian_train_data.plot.scatter(
    title="MultiCons Approach 1",
    ax=axes[0, 0],
    c=consensus_1.labels_,
    **common_kwargs
)
gaussian_train_data.plot.scatter(
    title="MultiCons Approach 2",
    ax=axes[0, 1],
    c=consensus_2.labels_,
    **common_kwargs
)
gaussian_train_data.plot.scatter(
    title="MultiCons Approach 3",
    ax=axes[1, 0],
    c=consensus_3.labels_,
    **common_kwargs
)
gaussian_train_data.plot.scatter(
    title="MultiCons Approach 4",
    ax=axes[1, 1],
    c=consensus_4.labels_,
    **common_kwargs
)
gaussian_train_data.plot.scatter(
    title="MultiCons Approach 5",
    ax=axes[2, 1],
    c=consensus_5.labels_,
    **common_kwargs
)
fig.show()

# %% [markdown]
# **Also, let's visualize the ConsTrees:**

# %%
cons_tree = consensus_1.cons_tree()
# Save the ConsTree graph to a file
cons_tree.render(outfile=f"{file_prefix}GaussianConsTree1.svg", cleanup=True)
cons_tree

# %% [markdown]
# View ConsTree graph 1 in full size: [GaussianConsTree1.svg](../GaussianConsTree1.svg)

# %%
cons_tree = consensus_2.cons_tree()
# Save the ConsTree graph to a file
cons_tree.render(outfile=f"{file_prefix}GaussianConsTree2.svg", cleanup=True)
cons_tree

# %% [markdown]
# View ConsTree graph 2 in full size: [GaussianConsTree2.svg](../GaussianConsTree2.svg)

# %%
cons_tree = consensus_3.cons_tree()
# Save the ConsTree graph to a file
cons_tree.render(outfile=f"{file_prefix}GaussianConsTree3.svg", cleanup=True)
cons_tree

# %% [markdown]
# View ConsTree graph 1 in full size: [GaussianConsTree3.svg](../GaussianConsTree3.svg)

# %%
cons_tree = consensus_4.cons_tree()
# Save the ConsTree graph to a file
cons_tree.render(outfile=f"{file_prefix}GaussianConsTree4.svg", cleanup=True)
cons_tree

# %% [markdown]
# View ConsTree graph 4 in full size: [GaussianConsTree4.svg](../GaussianConsTree4.svg)

# %%
cons_tree = consensus_5.cons_tree()
# Save the ConsTree graph to a file
cons_tree.render(outfile=f"{file_prefix}GaussianConsTree5.svg", cleanup=True)
cons_tree

# %% [markdown]
# View ConsTree graph 5 in full size: [GaussianConsTree5.svg](../GaussianConsTree5.svg)

# %% [markdown]
# **Finally, let's compare the clustering results:**

# %%
true_labels = gaussian_distributions["class"].to_numpy()
# We compare the results using the pair-wise Jaccard Similarity Measure
pd.DataFrame(
    [
        ["K-means", jaccard_similarity(base_clusterings[0], true_labels)],
        ["Average linkage", jaccard_similarity(base_clusterings[1], true_labels)],
        ["Gaussian model", jaccard_similarity(base_clusterings[2], true_labels)],
        ["C-means", jaccard_similarity(base_clusterings[3], true_labels)],
        ["PAM", jaccard_similarity(base_clusterings[4], true_labels)],
        ["BIRCH", jaccard_similarity(base_clusterings[5], true_labels)],
        ["MultiCons_1", jaccard_similarity(consensus_1.labels_, true_labels)],
        ["MultiCons_2", jaccard_similarity(consensus_2.labels_, true_labels)],
        ["MultiCons_3", jaccard_similarity(consensus_3.labels_, true_labels)],
        ["MultiCons_4", jaccard_similarity(consensus_4.labels_, true_labels)],
        ["MultiCons_5", jaccard_similarity(consensus_5.labels_, true_labels)],
    ],
    columns=["Algorithm", "Jaccard"]
)
