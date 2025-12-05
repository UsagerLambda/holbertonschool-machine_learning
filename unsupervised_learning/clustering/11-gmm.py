#!/usr/bin/env python3
"""Calculate a GMM from a dataset."""

import sklearn.mixture


def gmm(X, k):
    """
    Calculate a Gaussian Mixture Model from a dataset.

    Args:
        X (np.ndarray): of shape (n, d) containing the dataset
        k (int): number of clusters

    Returns:
        tuple: (pi, m, S, clss, bic) where:
            - pi is a np.ndarray of shape (k,) containing the cluster priors
            - m is a np.ndarray of shape (k, d) containing the centroid means
            - S is a np.ndarray of shape (k, d, d) containing the covariance
              matrices
            - clss is a np.ndarray of shape (n,) containing the cluster
              indices for each data point
            - bic is a np.ndarray containing the BIC value for the model
    """
    gmm_models = sklearn.mixture.GaussianMixture(k).fit(X)
    pi = gmm_models.weights_
    m = gmm_models.means_
    S = gmm_models.covariances_
    clss = gmm_models.predict(X)
    bic = gmm_models.bic(X)
    return pi, m, S, clss, bic
