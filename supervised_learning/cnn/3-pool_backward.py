#!/usr/bin/env python3
"""Effectue la propagation arrière d'une couche de pooling."""


import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """
    Effectue la rétropropagation sur une couche de pooling.

    Arguments :
        dA : numpy.ndarray de forme (m, h_new, w_new, c)
            Contient les dérivées partielles par rapport
                à la sortie de la couche de pooling
            m : nombre d'exemples
            h_new : hauteur de la sortie
            w_new : largeur de la sortie
            c : nombre de canaux
        A_prev : numpy.ndarray de forme (m, h_prev, w_prev, c)
            Contient la sortie de la couche précédente
            h_prev : hauteur de la couche précédente
            w_prev : largeur de la couche précédente
            c : nombre de canaux
        kernel_shape : tuple (kh, kw) taille du noyau pour le pooling
            kh : hauteur du noyau
            kw : largeur du noyau
        stride : tuple (sh, sw) pas pour le pooling
            sh : pas pour la hauteur
            sw : pas pour la largeur

    Retourne :
        dA_prev : dérivées partielles par rapport à la couche précédente
    """
    m, h_new, w_new, c = dA.shape
    kh, kw = kernel_shape
    sh, sw = stride
    dA_prev = np.zeros_like(A_prev)

    for i in range(m):
        for h in range(h_new):
            for w in range(w_new):
                for ch in range(c):
                    row = h * sh
                    col = w * sw

                    region = A_prev[i, row:row+kh, col:col+kw, ch]
                    if mode == 'max':
                        # Récupère la valeur max de la région
                        mask = (region == np.max(region))
                        # Stocke à la position du mask le gradient de dA dans
                        # dA_prev
                        dA_prev[i, row:row+kh, col:col+kw, ch] += (
                            mask * dA[i, h, w, ch]
                        )
                    elif mode == 'avg':
                        # Divise le gradient reçu par le nombre de pixels
                        # du kernel
                        da = dA[i, h, w, ch] / (kh * kw)
                        # Chaque pixel de la région reçoit une part égale
                        # du gradient
                        dA_prev[i, row:row + kh, col:col + kw, ch] += (
                            np.ones((kh, kw)) * da
                        )
    return dA_prev
