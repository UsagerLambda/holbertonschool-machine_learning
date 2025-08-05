#!/usr/bin/env python3
"""
Function that performs matrix multiplication between two given matrices.
"""
import numpy as np


def np_matmul(mat1, mat2):
    """nap_matmul

    Args:
        mat1 (np.array): The first matrice
        mat2 (np.array): The second matrice

    Returns:
        np.array: New NumPy array resulting from the matrix multiplication
        of mat1 and mat2.
    """
    return np.matmul(mat1, mat2)  # ou mat1 @ mat2
