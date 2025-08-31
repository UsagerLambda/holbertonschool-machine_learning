#!/usr/bin/env python3
"""Initialize Keras model using a Adam optimization."""

import tensorflow.keras as K


def one_hot(labels, classes=None):
    """
    Encode une liste de labels en représentation one-hot.

    Args:
        labels (list | np.ndarray | tf.Tensor):
            Liste ou tableau de labels entiers (ex: [0, 1, 2]).
            Peut aussi être un tenseur TensorFlow.
        classes (int, optional):
            Nombre total de classes (longueur des vecteurs one-hot).
            Si None, sera déduit automatiquement comme max(labels) + 1.

    Returns:
        tf.Tensor:
            Tenseur TensorFlow de forme (n_samples, classes) contenant
            l'encodage one-hot des labels.
    """
    if classes is None:
        classes = max(labels) + 1  # Trouve le nombre de classes
    # Créer un objet qui sait transformer des entiers en vecteur one-hot
    # num_tokens = combien de colonnes (classes)
    # output_mode = indique le type d'encodage
    layer = K.layers.CategoryEncoding(num_tokens=classes,
                                      output_mode="one_hot")
    # Utilise l'objet avec les données (labels),
    return layer(labels)
