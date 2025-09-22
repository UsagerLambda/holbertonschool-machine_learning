#!/usr/bin/env python3
"""Convolution valide sur des images en niveaux de gris."""

import numpy as np


def convolve_grayscale_same(images, kernel):
    """Effectue une convolution valide sur des images en niveaux de gris.

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

    Returns:
        numpy.ndarray: contenant les images convoluées
    """
    # On récupère la taille des images et leurs nombre
    m, h, w = images.shape
    # On récupère la taille du filtre
    kh, kw = kernel.shape

    pad_h = (kh - 1) // 2
    pad_w = (kw - 1) // 2

    # On crée un tableau 3D vide de taille y = oh, x = ow
    # et z = m (nombres images)
    output = np.zeros((m, h, w))

    # Rajoute (pour kernel = 3x3) 1 ligne en bas et en haut
    # ainsi que une colonne à gauche et à droite avec la valeur 0.
    padded_images = np.pad(images,
                          ((0, 0), (pad_h, pad_h), (pad_w, pad_w)),
                          mode='constant', constant_values=0)

    # Parcourt les lignes
    for i in range(h):
        # Parcourt les colonnes
        for j in range(w):
            # Récupère les valeurs dans un carré
            # de taille kernel dans les images depuis les index i et j
            sliced = padded_images[:, i:i+kh, j:j+kw]
            # fait la sommes des valeurs des axes 1 et 2 (hauteurs et largeur)
            # de la multiplication des index sliced * kernel
            sumed = np.sum(sliced * kernel, axis=(1, 2))
            # Stocke cette somme à l'index de chaque tableau
            output[:, i, j] = sumed
    return output
