#!/usr/bin/env python3
"""Calculates the F1 score of each class for a confusion matrix."""

import numpy as np
sensitivity = __import__('1-sensitivity').sensitivity
precision = __import__('2-precision').precision


def f1_score(confusion):
    """
    Calculate the F1 score of each class for a confusion matrix.

    Args:
        confusion: a confusion numpy.ndarray of shape (classes, classes)
                  where row indices represent the correct labels and
                  column indices represent the predicted labels
        classes: the number of classes

    Returns:
        numpy.ndarray of shape (classes,) containing the F1 score of each class

    Note:
        You must use sensitivity = __import__('1-sensitivity').sensitivity and
        precision = __import__('2-precision').precision create previously
    """
    prec = precision(confusion)
    sens = sensitivity(confusion)
    return 2 * (prec * sens) / (prec + sens)
