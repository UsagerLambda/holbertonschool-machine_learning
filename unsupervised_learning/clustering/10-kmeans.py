#!/usr/bin/env python3
"""Perform K-means on a dataset."""

import sklearn.cluster


def kmeans(X, k):
    """
    Perform K-means on a dataset.

    Args:
        X (numpy.ndarray): of shape (n, d) containing the dataset
        k (int): number of clusters

    Returns:
        _type_: _description_
    """
    kmeans_model = sklearn.cluster.KMeans(k).fit(X)
    C = kmeans_model.cluster_centers_
    clss = kmeans_model.labels_
    return C, clss
