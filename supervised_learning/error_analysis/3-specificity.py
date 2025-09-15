#!/usr/bin/env python3
"""Calculate the specificity for each class in a multi-class classification."""

import numpy as np


def specificity(confusion):
    """Calculate the specificity for each class in multi-class classification.

    Args:
        confusion (numpy.ndarray): A confusion matrix of shape
            (classes, classes)
            where row indices represent the correct labels
            and column indices represent the predicted labels

    Returns:
        numpy.ndarray: A numpy array of shape (classes,) containing the
                      specificity of each class
    """
    nb_classes = confusion.shape[1]
    specificity = np.zeros((nb_classes,))
    for i in range(confusion.shape[0]):
        VP = confusion[i, i]
        FP = np.sum(confusion[:, i]) - VP
        FN = np.sum(confusion[i, :]) - VP
        total = np.sum(confusion)
        VN = total - VP - FP - FN
        specificity[i] = VN / (VN + FP)

    return specificity
