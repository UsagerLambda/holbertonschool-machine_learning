#!/usr/bin/env python3
"""Calculate the cost of a neural network with L2 regularization."""

import numpy as np
import tensorflow as tf


def l2_reg_cost(cost, lambtha, weights, L, m):
    """Calculate the cost of a neural network with L2 regularization.

    Args:
        cost (numpy_ndarray): cost of the network without L2 regularization.
        lambtha (float): regularization parameter.
        weights (dict): dict of weights and biases
            (numpy.ndarrays) of the neural network.
        L (int): number of layers in the neural network
        m (int): number of data points used

    Return:
        float: the cost of the network accounting for L2 regularization
    """
    l2_p = 0
    for i in range(1, L+1):
        W = weights[f"W{i}"]    # Récupère les poids de la couche
        l2_p += np.sum(W ** 2)  # Sommes des poids de la couche

    l2_p = cost + (lambtha / (2 * m)) * l2_p
    return l2_p
