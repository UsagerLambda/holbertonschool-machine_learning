#!/usr/bin/env python3
"""Propagation avant utilisant le Dropout."""

import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """Effectue la descente de gradient with dropout sur un réseau de neurones.

    Args:
        Y (numpy.ndarray): Tableau one-hot de forme (classes, m) contenant les
                          étiquettes correctes pour les données.
        weights (dict): Dictionnaire des poids et biais du réseau de neurones.
        cache (dict): Dictionnaire des sorties et masques de dropout de chaque
                     couche du réseau de neurones.
        alpha (float): Taux d'apprentissage.
        keep_prob (float): Probabilité qu'un nœud soit conservé.
        L (int): Nombre de couches du réseau.

    Note:
        classes est le nombre de classes
        m est le nombre de points de données
    """
    m = Y.shape[1]
    DZ = cache[f"A{L}"] - Y

    for i in range(L, 0, -1):
        A_prev = cache[f"A{i-1}"]  # Activation de la couche precedente

        DW = np.matmul(DZ, A_prev.T) / m  # Calcul gradient des poids + biais
        DB = np.sum(DZ, axis=1, keepdims=True) / m

        if i > 1:
            # Calcul du gradient par rapport à A_prev
            dA_prev = np.matmul(weights[f"W{i}"].T, DZ)

            # Application du masque de dropout
            D_prev = cache[f"D{i-1}"]
            dA_prev = dA_prev * D_prev
            dA_prev = dA_prev / keep_prob

            # Application de la dérivée de tanh
            derivative_A_prev = 1 - np.power(A_prev, 2)
            DZ = dA_prev * derivative_A_prev

        weights[f"W{i}"] = weights[f"W{i}"] - alpha * DW
        weights[f"b{i}"] = weights[f"b{i}"] - alpha * DB
