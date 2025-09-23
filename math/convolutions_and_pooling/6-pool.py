#!/usr/bin/env python3
"""Convolution valide sur des images."""

import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
    """Effectue un pooling sur des images.

    Args:
        images (numpy.ndarray): de forme (m, h, w, c) contenant plusieurs
                               images
                               - m est le nombre d'images
                               - h est la hauteur en pixels des images
                               - w est la largeur en pixels des images
                               - c est le nombre de canaux dans l'image
        kernel_shape (tuple): de forme (kh, kw) contenant la forme du noyau
                             pour le pooling
                             - kh est la hauteur du noyau
                             - kw est la largeur du noyau
        stride (tuple): de forme (sh, sw)
                       - sh est le stride pour la hauteur de l'image
                       - sw est le stride pour la largeur de l'image
        mode (str): indique le type de pooling
                   - 'max' indique un max pooling
                   - 'avg' indique un average pooling

    Returns:
        numpy.ndarray: contenant les images pooled
    """
    m, h, w, c = images.shape
    kh, kw = kernel_shape
    sh, sw = stride

    pad_h, pad_w = 0, 0  # C'est strided donc pas d'augmentation de résolution

    nh = ((h + 2 * pad_h - kh) // sh) + 1
    nw = ((w + 2 * pad_w - kw) // sw) + 1

    output = np.zeros((m, nh, nw, c))

    padded_images = np.pad(images, (
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
