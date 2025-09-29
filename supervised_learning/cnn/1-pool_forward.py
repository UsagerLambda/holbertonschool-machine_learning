#!/usr/bin/env python3
"""Effectue une propagation avant du pooling sur un tenseur d'entrée."""


import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Effectue une propagation avant du pooling sur un tenseur d'entrée.

    Args:
        A_prev (numpy.ndarray): tenseur d'entrée de forme
            (m, h_prev, w_prev, c_prev)
            contenant la sortie de la couche précédente.
            - m : nombre d'exemples
            - h_prev : hauteur de la couche précédente
            - w_prev : largeur de la couche précédente
            - c_prev : nombre de canaux de la couche précédente
        kernel_shape (tuple): (kh, kw) taille du noyau de pooling
            - kh : hauteur du noyau
            - kw : largeur du noyau
        stride (tuple): (sh, sw) strides du pooling
            - sh : stride vertical
            - sw : stride horizontal
        mode (str): 'max' ou 'avg', indique le type de pooling à appliquer

    Returns:
        numpy.ndarray: sortie de la couche de pooling
    """
    m, h, w, c = A_prev.shape
    kh, kw = kernel_shape
    sh, sw = stride

    pad_h, pad_w = 0, 0

    nh = ((h + 2 * pad_h - kh) // sh) + 1
    nw = ((w + 2 * pad_w - kw) // sw) + 1

    output = np.zeros((m, nh, nw, c))

    padded_images = np.pad(A_prev, (
        (0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)),
                           mode='constant', constant_values=0)

    for i in range(nh):
        for j in range(nw):
            row = i * sh
            col = j * sw
            sliced = padded_images[:, row:row+kh, col:col+kw]
            if mode == 'max':
                output[:, i, j, :] = np.max(sliced, axis=(1, 2))
            if mode == 'avg':
                output[:, i, j, :] = np.mean(sliced, axis=(1, 2))
    return output
