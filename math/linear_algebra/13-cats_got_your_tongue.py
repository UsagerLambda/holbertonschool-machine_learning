#!/usr/bin/env python3
"""
A function that concatenate 2 numpy arrays by rows or columns
based on the axis variable.
"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """np_cat

    Args:
        mat1 (np.array): The first matrix to concatenate.
        mat2 (np.array): The second matrix to concatenate.
        axis (int, optional): The axis along which the matrices
        will be concatenated.
        0 for rows concatenation (vertical).
        1 for columns concatenation (horizontal).

    Returns:
        np.ndarray: New NumPy array resulting from the concatenation of
        mat1 and mat2.
    """
    return np.concatenate((mat1, mat2), axis)
