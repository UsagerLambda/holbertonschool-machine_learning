#!/usr/bin/env python3
"""Method that train a model using mini-batch gradient descend."""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, verbose=True, shuffle=False):
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
        validation_data (tuple): contient les vraies réponses (labels)
            pour un ensemble d’exemples que le modèle n’a jamais vus pendant
            l’entraînement. À la fin de chaque époque, Keras compare les
            prédictions du modèle sur ces exemples avec les vraies réponses
            pour calculer la performance (perte, précision, etc.).
        early_stopping (bool): Indique si l'early stopping doit être utilisé.
        patience (int): patience used for early stopping
        verbose (bool, optional): Indique si la progression de l’entraînement
            doit être affichée. Par défaut True.
        shuffle (bool, optional): Indique si les lots doivent être mélangés
            à chaque époque. Par défaut False.


    Returns:
        keras.history: object generated after training the model.
    """
    callback =[]
    if validation_data:
        if early_stopping:
            callback.append(K.callbacks.EarlyStopping(
                monitor="val_loss", patience=patience))
        History = network.fit(
            data,
            labels,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose,
            shuffle=shuffle,
            validation_data=validation_data,
            callbacks=callback
            )
    else:
        History = network.fit(
            data,
            labels,
            batch_size=batch_size,
            epochs=epochs,
            verbose=verbose,
            shuffle=shuffle
            )

    return History
