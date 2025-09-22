#!/usr/bin/env python3
"""Convolution valide sur des images en niveaux de gris."""

import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
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
        padding (tuple): de forme (ph, pw)
                        - ph est le padding pour la hauteur de l'image
                        - pw est le padding pour la largeur de l'image
                        - l'image doit être paddée avec des 0

    Returns:
        numpy.ndarray: contenant les images convoluées
    """
    m, h, w = images.shape
    kh, kw = kernel.shape

    ph = padding[0]
    pw = padding[1]

    # Contrairement à l'exercice précedent ou l'ont réajustait les images
    # à leurs taille d'origine, ici nous augmenton la résolution d'origine
    # Donc nous devons réajuster la taille des images avant de parcourir
    # leurs matrices
    nh = h + 2 * ph - (kh - 1)
    nw = w + 2 * pw - (kw - 1)

    # Créer le tableau 3D de sortie ajusté à la nouvelle taille
    output = np.zeros((m, nh, nw))

    # Rajoute (pour kernel = 3x3) 1 ligne en bas et en haut
    # ainsi que une colonne à gauche et à droite avec la valeur 0.
    padded_images = np.pad(images, (
        (0, 0), (ph, ph), (pw, pw)),
                           mode='constant', constant_values=0)

    # Parcourt les lignes
    for i in range(nh):
        # Parcourt les colonnes
        for j in range(nw):
            sliced = padded_images[:, i:i+kh, j:j+kw]
            sumed = np.sum(sliced * kernel, axis=(1, 2))
            output[:, i, j] = sumed
    return output
