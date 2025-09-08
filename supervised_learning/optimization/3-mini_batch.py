#!/usr/bin/env python3
"""Create mini-batches."""

import numpy as np
import tensorflow as tf

shuffle_data = __import__('2-shuffle_data').shuffle_data


def create_mini_batches(X, Y, batch_size):
    """Create mini-batches to be used for training.

    Args:
        X (numpy.ndarray): matrix of shape (m, nx) representing input data
        Y (numpy.ndarray): matrix of shape (m, ny) representing labels
        batch_size (int): Number of data points in a batch
    """
    # Initialise une liste vide pour stocker les mini-batches.
    mini_batches = []
    # Mélange les données et les labels de façon identique.
    X_shuffle, Y_shuffle = shuffle_data(X, Y)
    # Boucle sur les indices de 0 à m (nombre d'exemples),
    # avec un pas de batch_size.
    for k in range(0, X.shape[0], batch_size):
        # Sélectionne les exemples de k à batch_size
        X_batch = X_shuffle[k: k + batch_size]
        Y_batch = Y_shuffle[k: k + batch_size]
        # Ajoute le tuple (X_batch, Y_batch) dans la liste mini_batches
        mini_batches.append((X_batch, Y_batch))
    return mini_batches
