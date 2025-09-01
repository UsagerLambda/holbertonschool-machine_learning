#!/usr/bin/env python3
"""Method that train a model using mini-batch gradient descend."""

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False,
                alpha=0.1, decay_rate=1, verbose=True, shuffle=False):
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

        learning_rate_decay (bool, optionnel): Indique si le taux d’apprentissage doit décroître
            au fil des époques. Si True, le taux d’apprentissage est mis à jour à chaque époque
            selon la formule : lr = alpha / (1 + decay_rate * epoch).
            Cela permet de commencer avec un taux d’apprentissage élevé puis de le réduire
            progressivement, ce qui aide à stabiliser la convergence du modèle.

        alpha (float, optionnel): Taux d’apprentissage initial utilisé si learning_rate_decay est True.

        decay_rate (float, optionnel): Facteur de décroissance du taux d’apprentissage.

        verbose (bool, optional): Indique si la progression de l’entraînement
            doit être affichée. Par défaut True.

        shuffle (bool, optional): Indique si les lots doivent être mélangés
            à chaque époque. Par défaut False.

    Returns:
        keras.history: object generated after training the model.
    """
    callback = []
    if validation_data:
        if early_stopping:
            callback.append(K.callbacks.EarlyStopping(
                monitor="val_loss", patience=patience))
        if learning_rate_decay:
            def scheduler(epoch):
                new_lr = alpha / (1 + decay_rate * epoch)
                print(f"\nEpoch {epoch+1}: LearningRateScheduler "
                      f"setting learning rate to {new_lr}.")
                return new_lr
            callback.append(K.callbacks.LearningRateScheduler(
                scheduler, verbose=0))
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
