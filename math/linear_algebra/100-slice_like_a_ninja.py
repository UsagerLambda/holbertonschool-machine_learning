#!/usr/bin/env python3
"""
Function that slice a matrix with specified parameter in axes
"""


def np_slice(matrix, axes={}):
    """np_slice

    Args:
        matrix (np.ndarray): The matrix to slice
        axes (dict): dictionnary where each key is an axis
                    and the value is a tuple representing the slice

    Returns:
        np.ndarray: The sliced matrix
    """
    # Crée une liste contenant un slice(None)
    # pour chaque dimension de la matrice
    slices = [slice(None)] * matrix.ndim
    for axis, value in axes.items():
        # Remplace le slice[None] à l’axe donné
        # par un nouveau slice construit à partir de value
        slices[axis] = slice(*value)
    # Applique le slicing en transformant la liste de slices en tuple
    return matrix[tuple(slices)]
