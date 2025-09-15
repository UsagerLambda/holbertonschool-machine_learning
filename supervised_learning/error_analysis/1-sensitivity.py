#!/usr/bin/env python3

import numpy as np


def sensitivity(confusion):
    """Calculate the sensitivity for each class in a confusion matrix.

    Args:
        confusion (numpy.ndarray): Matrix of shape (classes, classes)
            where row indices represent the correct labels and column
            indices represent the predicted labels

    Return:
        numpy.ndarray: Matrix of shape (classes,) containing the
            sensitivity of each class
    """
    nb_classes = confusion.shape[1]
    sensitivity = np.zeros((nb_classes,))
    for i in range(confusion.shape[0]):      # pour chaque ligne
        TP = confusion[i, i]
        sum = 0
        for j in range(confusion.shape[1]):  # pour chaque colonne
            sum += confusion[i, j]
        sensitivity[i] = TP / sum
    return sensitivity
