#!/usr/bin/env python3
"""Method that train a model using mini-batch gradient descend."""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                verbose=True, shuffle=False):
    """Train a model using mini-batch gradient descend.

    Args:
        network (keras.Model): Modèle à entraîner.
        data (numpy.ndarray): Données d’entrée de forme (m, nx),
            où m est le nombre d’exemples et nx le nombre de caractéristiques.
        labels (numpy.ndarray): Tableau one-hot de forme (m, classes)
            contenant les étiquettes associées aux données.
        batch_size (int): Taille des lots utilisés pour l’apprentissage
            par descente de gradient en mini-batch.
        epochs (int): Nombre de passages complets sur les données
            pendant l’entraînement.
        verbose (bool, optional): Indique si la progression de l’entraînement
            doit être affichée. Par défaut True.
        shuffle (bool, optional): Indique si les lots doivent être mélangés
            à chaque époque. Par défaut False.


    Returns:
        keras.history: object generated after training the model.
    """
    History = network.fit(
        data,
        labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle
        )

    return History
