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
    elif padding == 'same':
        # (h-1)*sh = position du dernier pixel intégré du kernel
        # dans l'axe visé
        # Exemple: si h=28 et stride=2, le kernel n'ira pas en 28 car il se
        # déplace de 2 en 2 et 29 n'existe pas
        # + kh = ajoute la taille du kernel (aire de kh x kw)
        # - h = retire la taille originale pour le padding total
        # → // 2 = divise par 2 pour chaque côté opposé
        pad_h = ((h - 1) * sh + kh - h) // 2
        pad_w = ((w - 1) * sw + kw - w) // 2
    elif padding == 'valid':
        pad_h, pad_w = 0, 0

    # h = taille initiale de l'image
    # 2 * pad_h/w ajoute les pixels fictifs du padding
    # (x2 pour chaque côté opposé)
    # -kh/w retire la taille du kernel, car il doit rester dans l'image
    # → // sh/w (divisé par le stride) nombre de déplacements possibles
    # + 1 (ajoute la position initiale)
    nh = ((h + 2 * pad_h - kh) // sh) + 1
    nw = ((w + 2 * pad_w - kw) // sw) + 1

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
