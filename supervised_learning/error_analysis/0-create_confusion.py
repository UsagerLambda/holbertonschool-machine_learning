#!/usr/bin/env python3
"""Create confusion matrix."""

import numpy as np


def create_confusion_matrix(labels, logits):
    """
    Crée une matrice de confusion pour l'analyse des erreurs de classification.

    Args:
        labels (numpy.ndarray): Tableau one-hot de forme (m, classes)
            contenant les vraies étiquettes.
            m : nombre d'échantillons
            classes : nombre de classes
        logits (numpy.ndarray): Tableau one-hot de forme (m, classes)
            contenant les étiquettes prédites.

    Returns:
        numpy.ndarray: Matrice de confusion de forme (classes, classes)
            où les lignes représentent les vraies étiquettes
            et les colonnes les étiquettes prédites.
    """
    # Récupère le nombre de classes à partir de la dimension des labels
    nb_classes = labels.shape[1]

    # Convertit les encodages one-hot en indices de classes
    y_true = np.argmax(labels, axis=1)  # Vraies classes
    y_pred = np.argmax(logits, axis=1)  # Classes prédites

    confusion = np.zeros((nb_classes, nb_classes))

    # Pour chaque échantillon, incrémente la case correspondante dans la matrix
    # confusion[i][j] = nb d'échantillons de classe i prédits comme classe j
    for true_class, pred_class in zip(y_true, y_pred):
        confusion[true_class][pred_class] += 1

    return confusion
