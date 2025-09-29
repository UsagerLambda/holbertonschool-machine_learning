#!/usr/bin/env python3
"""Réalise la propagation avant d'une couche de convolution."""


import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """
    Effectue la propagation avant d'une couche de convolution.

    Args:
        A_prev (numpy.ndarray): tableau de forme (m, h_prev, w_prev, c_prev)
                contenant la sortie de la couche précédente
            - m : nombre d'exemples
            - h_prev : hauteur de la couche précédente
            - w_prev : largeur de la couche précédente
            - c_prev : nombre de canaux de la couche précédente
        W (numpy.ndarray): tableau de forme (kh, kw, c_prev, c_new)
                contenant les kernels pour la convolution
            - kh : hauteur du filtre
            - kw : largeur du filtre
            - c_prev : nombre de canaux de la couche précédente
            - c_new : nombre de canaux de la sortie
        b (numpy.ndarray): tableau de forme (1, 1, 1, c_new)
                contenant les biais appliqués à la convolution
        activation (fonction): fonction d'activation appliquée à la convolution
        padding (str, optionnel): 'same' ou 'valid',
                indique le type de padding utilisé (défaut : "same")
        stride (tuple, optionnel): (sh, sw),
                contenant les strides pour la convolution (défaut : (1, 1))
            - sh : stride pour la hauteur
            - sw : stride pour la largeur

    Returns:
        numpy.ndarray: sortie de la couche de convolution
    """
    m, h_prev, w_prev, c_prev = A_prev.s_prevape
    kh, kw, kc, c_new = W.shape
    sh, sw = stride
    kc = c_prev

    if isinstance(padding, tuple):
        pad_h, pad_w = padding
    elif padding == 'same':
        pad_h = ((h_prev - 1) * sh + kh - h_prev) // 2
        pad_w = ((w_prev - 1) * sw + kw - w_prev) // 2
    elif padding == 'valid':
        pad_h, pad_w = 0, 0

    nh = ((h_prev + 2 * pad_h - kh) // sh) + 1
    nw = ((w_prev + 2 * pad_w - kw) // sw) + 1

    output = np.zeros((m, nh, nw, c_new))

    padded_images = np.pad(A_prev, (
        (0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)),
                           mode='constant', constant_values=0)

    for i in range(nh):
        for j in range(nw):
            for k in range(c_new):
                row = i * sh
                col = j * sw
                sliced = padded_images[:, row:row+kh, col:col+kw, :]
                output[:, i, j, k] = np.sum(
                    sliced * W[:, :, :, k], axis=(1, 2, 3))
    return activation(output + b)
