#!/usr/bin/env python3
"""Propagation avant utilisant le Dropout."""

import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """Effectue la propagation avant en utilisant le Dropout.

    Args:
        X (numpy.ndarray): données d'entrée de forme (nx, m)
        weights (dict): poids et biais du réseau de neurones
        L (int): nombre de couches dans le réseau
        keep_prob (float): probabilité qu'un noeud soit conservé

    Returns:
        dict: sorties de chaque couche et masques dropout utilisés
    """
    cache = {}
    cache["A0"] = X

    for i in range(L):
        # Transformation linéaire : Z = W*A + b
        Z = np.matmul(weights[f"W{i+1}"], cache[f"A{i}"]) + weights[f"b{i+1}"]

        if i < L - 1:
            # Couches cachées : activation tanh + dropout
            A = np.tanh(Z)

            # Créer le masque dropout et appliquer l'inverted dropout
            D = (np.random.rand(*A.shape) < keep_prob).astype(int)
            A = (A * D) / keep_prob
            cache[f"D{i+1}"] = D
        else:
            # Couche de sortie : activation softmax (pas de dropout)
            expZ = np.exp(Z - np.max(Z, axis=0, keepdims=True))
            A = expZ / np.sum(expZ, axis=0, keepdims=True)

        cache[f"A{i+1}"] = A

    return cache
