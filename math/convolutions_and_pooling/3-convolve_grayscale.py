#!/usr/bin/env python3
"""Convolution valide sur des images en niveaux de gris."""

import numpy as np


def convolve_grayscale(images, kernel, padding='same', stride=(1, 1)):
    """Effectue une convolution sur des images en niveaux de gris.

    Args:
        images (numpy.ndarray): de forme (m, h, w) contenant plusieurs
                               images en niveaux de gris
                               - m est le nombre d'images
                               - h est la hauteur en pixels des images
                               - w est la largeur en pixels des images
        kernel (numpy.ndarray): de forme (kh, kw) contenant le noyau
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
    m, h, w = images.shape
    kh, kw = kernel.shape
    sh, sw = stride

    if isinstance(padding, tuple):
        pad_h, pad_w = padding
        nh = h + 2 * pad_h - (kh - 1)
        nw = w + 2 * pad_w - (kw - 1)
    elif padding == 'same':
        pad_h = kh // 2
        pad_w = kw // 2
        nh = h
        nw = w
    elif padding == 'valid':
        pad_h = pad_w = 0
        nh = (h - kh) // sh + 1
        nw = (w - kw) // sw + 1

    output = np.zeros((m, nh, nw))

    padded_images = np.pad(images, (
        (0, 0), (pad_h, pad_h), (pad_w, pad_w)),
                           mode='constant', constant_values=0)

    for i in range(nh):
        for j in range(nw):
            row = i * sh  # Calc l'index de départ en hauteur avec le stride sh
            col = j * sw  # Calc l'index de départ en largeur avec le stride sw
            # Extrait la fenêtre de convolution de taille (kh, kw)
            # à partir de row et col
            sliced = padded_images[:, row:row+kh, col:col+kw]
            output[:, i, j] = np.sum(sliced * kernel, axis=(1, 2))
    return output
