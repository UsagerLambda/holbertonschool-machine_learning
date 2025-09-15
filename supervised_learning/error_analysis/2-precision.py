#!/usr/bin/env python3
"""Calculate the precision for each class in a confusion matrix."""

import numpy as np


def precision(confusion):
    """
    Calculate the precision for each class in a confusion matrix.

    Args:
        confusion (numpy.ndarray): A confusion matrix of shape
            (classes, classes)
            where row indices represent the correct labels
            and column indices represent the predicted labels
        classes (int): The number of classes

    Returns:
        numpy.ndarray: A numpy array of shape (classes,)
            containing the precision of each class
    """
    nb_classes = confusion.shape[1]
    precision = np.zeros((nb_classes,))
    for i in range(confusion.shape[0]):
        TP = confusion[i, i]
        sum = np.sum(confusion[:, i])
        precision[i] = TP / sum
    return precision
