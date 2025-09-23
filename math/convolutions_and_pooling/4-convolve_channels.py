#!/usr/bin/env python3
"""Convolution valide sur des images."""

import numpy as np


def convolve_channels(images, kernel, padding='same', stride=(1, 1)):
    """Effectue une convolution sur des images avec canaux.

    Args:
        images (numpy.ndarray): de forme (m, h, w, c) contenant plusieurs
                               images
                               - m est le nombre d'images
                               - h est la hauteur en pixels des images
                               - w est la largeur en pixels des images
                               - c est le nombre de canaux dans l'image
        kernel (numpy.ndarray): de forme (kh, kw, c) contenant le noyau
                               pour la convolution
                               - kh est la hauteur du noyau
                               - kw est la largeur du noyau
        padding (tuple ou str): soit un tuple de (ph, pw), 'same', ou 'valid'
                               - si 'same', effectue une convolution same
                               - si 'valid', effectue une convolution valid
                               - si un tuple :
                                 - ph est le padding pour la hauteur de l'image
                                 - pw est le padding pour la largeur de l'image
                               - l'image doit être paddée avec des 0
        stride (tuple): de forme (sh, sw)
                       - sh est le stride pour la hauteur de l'image
                       - sw est le stride pour la largeur de l'image

    Returns:
        numpy.ndarray: contenant les images convoluées
    """
    m, h, w, c = images.shape
    kh, kw, kc = kernel.shape
    sh, sw = stride
    kc = c  # channel du kernel égal au nombre de channel

    if isinstance(padding, tuple):
        pad_h, pad_w = padding
    elif padding == 'same':
        pad_h = ((h - 1) * sh + kh - h) // 2 + 1
        pad_w = ((w - 1) * sw + kw - w) // 2 + 1
    elif padding == 'valid':
        pad_h, pad_w = 0, 0

    nh = ((h + 2 * pad_h - kh) // sh) + 1
    nw = ((w + 2 * pad_w - kw) // sw) + 1

    output = np.zeros((m, nh, nw))

    # (0, 0) dit que ont ne dois rien modifier dans l'axe 4
    # np.pad exige de le spécifier dans ses paramêtres
    padded_images = np.pad(images, (
        (0, 0), (pad_h, pad_h), (pad_w, pad_w), (0, 0)),
                           mode='constant', constant_values=0)

    for i in range(nh):
        for j in range(nw):
            row = i * sh
            col = j * sw
            sliced = padded_images[:, row:row+kh, col:col+kw, :]
            output[:, i, j] = np.sum(sliced * kernel, axis=(1, 2, 3))
    return output
