#!/usr/bin/env python3
"""
Function that transpose a given matrice
"""
import numpy as np


def np_transpose(matrix):
    """np_transpose

    Args:
        matrix (list): list of lists of integers

    Returns:
        np.array: the transposed given matrice
    """
    new = np.transpose(matrix)
    return new
