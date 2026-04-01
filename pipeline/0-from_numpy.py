#!/usr/bin/env python3

"""Convertit numpy array en DataFrame pandas avec des colonnes étiquetées."""

import pandas as pd
import string


def from_numpy(array):
    """Convertit numpy array en DataFrame pandas avec des colonnes étiquetées.

    Args:
        array: tableau numpy 2D

    Returns:
        DataFrame pandas avec les colonnes nommées alphabétiquement
    """
    cols = list(string.ascii_uppercase[:array.shape[1]])
    return pd.DataFrame(array, columns=cols)
