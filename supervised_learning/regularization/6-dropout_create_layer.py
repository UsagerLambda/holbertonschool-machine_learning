#!/usr/bin/env python3
"""Crée une couche avec régularisation dropout."""

import numpy as np
import tensorflow as tf


def dropout_create_layer(prev, n, activation, keep_prob, training=True):
    """Crée une couche avec régularisation dropout.

    Args:
        prev: un tenseur contenant la sortie de la couche précédente
        n: le nombre de nœuds que la nouvelle couche doit contenir
        activation: la fonction d'activation pour la nouvelle couche
        keep_prob: la probabilité qu'un nœud soit conservé
        training (bool, optional): un booléen indiquant si le modèle
            est en mode d'entraînement. Par défaut True.

    Returns:
        la sortie de la nouvelle couche
    """
    initializer = tf.keras.initializers.VarianceScaling(
        scale=2.0, mode=("fan_avg"))

    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_initializer=initializer,
    )
    dropout_layer = tf.keras.layers.Dropout(rate=1-keep_prob)

    dense_output = layer(prev)
    return dropout_layer(dense_output, training=training)
