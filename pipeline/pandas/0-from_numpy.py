#!/usr/bin/env python3

"""Convertit numpy array en DataFrame pandas avec des colonnes étiquetées."""

import pandas as pd


def from_numpy(array):
    """Convertit numpy array en DataFrame pandas avec des colonnes étiquetées.

    Args:
        array: tableau numpy 2D

    Returns:
        DataFrame pandas avec les colonnes nommées alphabétiquement
    """
    cols = [chr(65 + i) for i in range(array.shape[1])]
    return pd.DataFrame(array, columns=cols)
