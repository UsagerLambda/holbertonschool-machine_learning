#!/usr/bin/env python3
"""Encodage positionnel pour les modèles Transformer."""

import numpy as np


def positional_encoding(max_seq_len, dm):
    """
    Calculate l'encodage positionnel pour un transformer.

    Args:
        max_seq_len: Longueur maximale de la séquence.
        dm: Profondeur du modèle (dimensionnalité de l'embedding).

    Returns:
        numpy.ndarray: Tableau de forme (max_seq_len, dm) contenant
            les vecteurs d'encodage positionnel.
    """
    pe = np.zeros((max_seq_len, dm))
    for pos in range(max_seq_len):
        for i in range(0, dm, 2):
            pe[pos, i] = np.sin(pos / (10000 ** (2 * i / dm)))
            if i + 1 < dm:
                pe[pos, i + 1] = np.cos(pos / (10000 ** (2 * i / dm)))
    return pe
