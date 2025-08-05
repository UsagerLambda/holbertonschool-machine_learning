#!/usr/bin/env python3
"""
Function that transpose a given matrice
"""


def np_transpose(matrix):
    """np_transpose

    Args:
        matrix (list): list of lists of integers

    Returns:
        np.array: the transposed given matrice
    """
    new = matrix.T
    return new
