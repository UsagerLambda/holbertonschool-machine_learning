#!/usr/bin/env python3
"""Perform an agglomerative clustering on a dataset."""

import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """
    Perform an agglomerative clustering on a dataset.

    Args:
        X (np.ndarray): of shape (n, d) containing the dataset
        dist (int): the maximum cophenetic distance for all clusters

    Returns:
        clss: np.ndarray of shape (n,) containing the cluster indices
            for each datapoint
    """
    linkage = scipy.cluster.hierarchy.linkage(X, method='ward')
    scipy.cluster.hierarchy.dendrogram(linkage, color_threshold=dist)
    plt.show()
    # Coupe l'arbre à la hauteur dist en utlisant
    # 'distance' comme critère de coupe (distance = distance cophenetic)
    clss = scipy.cluster.hierarchy.fcluster(
        linkage, dist, criterion='distance')
    return clss
