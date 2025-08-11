#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt


def line():

    y = np.arange(0, 11) ** 3
    plt.figure(figsize=(6.4, 4.8))

    x = np.arange(0, 11)  # x allant de 0 à 10
    plt.plot(x, y, 'r-')  # Trace y en fonction de x, avec une ligne rouge
    plt.xlim(0, 10)  # Définit la limite de l'axe X entre 0 et 10
    plt.show()  # Affiche le graph
