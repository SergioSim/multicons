# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.8
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
from matplotlib import pyplot as plt
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans, SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn_extra.cluster import KMedoids

from multicons import MultiCons

np.set_printoptions(threshold=100)

# %%
# Load the data
prefix = "" if path.exists("cassini.csv") else "docs/"
file_name = f"{prefix}cassini.csv"
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
fcm = FCM(n_clusters=3)
fcm.fit(cassini_train_data.values)
base_clusterings.append(fcm.predict(cassini_train_data.values))
cassini_train_data.plot.scatter(
    title="C-means", ax=axes[1, 1], c=base_clusterings[-1], **common_kwargs
)

# PAM
base_clusterings.append(KMedoids(n_clusters=3).fit_predict(cassini_train_data))
cassini_train_data.plot.scatter(
    title="PAM", ax=axes[1, 2], c=base_clusterings[-1], **common_kwargs
)

# Spectral
base_clusterings.append(
    SpectralClustering(n_clusters=3).fit_predict(cassini_train_data)
)
cassini_train_data.plot.scatter(
    title="Spectral", ax=axes[2, 0], c=base_clusterings[-1], **common_kwargs
)

# DBSCAN
base_clusterings.append(DBSCAN(eps=0.2).fit_predict(cassini_train_data))
cassini_train_data.plot.scatter(
    title="DBSCAN", ax=axes[2, 1], c=base_clusterings[-1], **common_kwargs
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
# Save the graph to a file
cons_tree.render(outfile=f"{prefix}CassiniConsTree.svg", cleanup=True)

# %% [markdown]
# The graph in full size: [CassiniConsTree.svg](../CassiniConsTree.svg)
