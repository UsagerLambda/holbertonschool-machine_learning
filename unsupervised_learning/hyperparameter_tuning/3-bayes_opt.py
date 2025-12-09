#!/usr/bin/env python3
"""Init a Bayesian optimization class."""

import numpy as np
GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """
    Représente l'optimisation bayésienne sur une fonction boîte noire.
    """
    def __init__(self, f, X_init, Y_init,
                 bounds, ac_samples, l=1,
                 sigma_f=1, xsi=0.01, minimize=True):
        """
        Initialise l'optimisation bayésienne.

        Args:
            f (function): la fonction boîte noire à optimiser
            X_init (np.ndarray): de forme (t, 1) représentant
                les entrées déjà échantillonnées avec la fonction boîte noire
            Y_init (np.ndarray): de forme (t, 1) représentant
                les sorties de la fonction boîte noire pour chaque
                entrée dans X_init
            bounds (tuple): de (min, max) représentant les limites de
                l'espace de recherche dans une dimension donnée
            ac_samples (int): le nombre d'échantillons à analyser
                pendant la fonction d'acquisition
            l (int, optional): paramètre de longueur pour le noyau.
            sigma_f (int, optional): l'écart type donné à la sortie
                de la fonction boîte noire. Defaults to 1.
            xsi (float, optional): le paramètre d'exploration-exploitation
                pour la fonction d'acquisition. Defaults to 0.01.
            minimize (bool, optional): détermine si l'optimisation
                doit être effectuée pour minimisation (True) ou
                maximisation (False). Defaults to True.
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        x_min, x_max = bounds
        self.X_s = np.linspace(x_min, x_max, ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize
