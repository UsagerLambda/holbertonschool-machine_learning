#!/usr/bin/env python3
"""Effectue la propagation arrière d'une couche de convolution."""


import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """
    Effectue la propagation arrière d'une couche de convolution.

    Arguments :
        dZ : numpy.ndarray de forme (m, h_new, w_new, c_new)
            Gradient de la couche suivante (précédente dans le sens
            de la back prop)
            m : nombre d'exemples
            h_new : hauteur de la sortie
            w_new : largeur de la sortie
            c_new : nombre de canaux de la sortie
        A_prev : numpy.ndarray de forme (m, h_prev, w_prev, c_prev)
            Sortie de la couche précédente
            h_prev : hauteur de la couche précédente
            w_prev : largeur de la couche précédente
            c_prev : nombre de canaux de la couche précédente
        W : numpy.ndarray de forme (kh, kw, c_prev, c_new)
            Kernel pour la convolution
            kh : hauteur du filtre
            kw : largeur du filtre
        b : numpy.ndarray de forme (1, 1, 1, c_new)
            Biais appliqués à la convolution
        padding : str, optionnel
            Type de padding utilisé ("same" ou "valid"), par défaut "same"
        stride : tuple (sh, sw), optionnel
            Pas de la convolution pour la hauteur (sh) et la largeur (sw),
            par défaut (1, 1)

    Retourne :
        dA_prev : dérivées partielles par rapport à l'entrée précédente
        dW : dérivées partielles par rapport aux noyaux
        db : dérivées partielles par rapport aux biais
    """
    m, h_new, w_new, c_new = dZ.shape
    m, h_prev, w_prev, c_prev = A_prev.shape
    kh, kw, kc, c_new = W.shape
    sh, sw = stride

    if isinstance(padding, tuple):
        pad_h, pad_w = padding
    elif padding == 'same':
        pad_h = ((h_prev - 1) * sh + kh - h_prev) // 2 + 1
        pad_w = ((w_prev - 1) * sw + kw - w_prev) // 2 + 1
    elif padding == 'valid':
        pad_h, pad_w = 0, 0

    dA_prev = np.zeros(A_prev.shape)  # Prev layer
    dW = np.zeros(W.shape)  # Kernel

    pad_A = np.pad(
        A_prev,
        ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)),
        mode='constant', constant_values=0
    )

    pad_dA = np.pad(
        dA_prev,
        ((0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)),
        mode='constant', constant_values=0
    )

    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)
    # sommes de toutes les erreurs

    for i in range(m):  # Pour chaque image du batch
        for h in range(h_new):  # Pour chaque ligne de sortie
            for w in range(w_new):  # Pour chaque colonne de sortie
                for c in range(c_new):  # Pour chaque canal de sortie
                    # Position de départ dans l'image paddée
                    row = h * sh
                    col = w * sw

                    # Extrait la region du kernel sur l'image
                    region = pad_A[i, row:row+kh, col:col+kw, :]

                    # Accumulation du gradient du filtre : chaque région qui a
                    # été convoluée contribue au gradient proportionnellement
                    # à l'erreur dZ à cette position
                    dW[:, :, :, c] += region * dZ[i, h, w, c]

                    # Distribution du gradient vers l'entrée : l'erreur dZ est
                    # redistribuée vers la région d'entrée qui a produit cette
                    # sortie, pondérée par les poids
                    pad_dA[i, row:row+kh, col:col+kw, :] += (
                        W[:, :, :, c] * dZ[i, h, w, c]
                    )

    # Réajuste la taille de l'image de la sortie précédente
    if pad_h > 0 and pad_w > 0:
        dA_prev = pad_dA[:, pad_h:-pad_h, pad_w:-pad_w, :]
    elif pad_h > 0:
        dA_prev = pad_dA[:, pad_h:-pad_h, :, :]
    elif pad_w > 0:
        dA_prev = pad_dA[:, :, pad_w:-pad_w, :]
    else:
        dA_prev = pad_dA
    return dA_prev, dW, db
