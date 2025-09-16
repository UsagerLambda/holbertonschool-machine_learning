#!/usr/bin/env python3
"""Module containing L2 regularization gradient descent implementation."""

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """
    Update the weights and biases of a neural network using gradient descent.

    This function applies gradient descent with L2 regularization to update
    the weights and biases of a neural network.

    Args:
        Y (numpy.ndarray): one-hot np.ndarray of shape (classes, m)
            that contains the correct labels of the data
            - classes : number of classes
            - m : number of datapoints
        weights (dict): dict of the weights and biases of the neural network
        cache (dict): dict of the outputs of each layer of the neural network
        alpha (float): learning rate
        lambtha (float): L2 regularization parameters
        L (int): number of layers of the network

    Returns:
        None: The function updates the weights dictionary in-place
    """
    m = Y.shape[1]
    DZ = cache[f"A{L}"] - Y

    for i in range(L, 0, -1):
        A_prev = cache[f"A{i-1}"]

        DW = np.matmul(DZ, A_prev.T) / m + (lambtha / m) * weights[f"W{i}"]
        DB = np.sum(DZ, axis=1, keepdims=True) / m

        if i > 1:
            DZ = np.matmul(weights[f"W{i}"].T, DZ) * (1 - A_prev * A_prev)

        weights[f"W{i}"] = weights[f"W{i}"] - alpha * DW
        weights[f"b{i}"] = weights[f"b{i}"] - alpha * DB
