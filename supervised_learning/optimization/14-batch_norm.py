#!/usr/bin/env python3
"""Create a normalization layer with batch_normalization."""

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """
    Crée une couche de normalisation par lot (batch normalization).

    Args:
        prev: sortie activée de la couche précédente
        n: nombre de nœuds dans la couche à créer
        activation: fonction d'activation à utiliser sur la sortie de la couche

    La couche doit utiliser tf.keras.layers.Dense comme couche
        de base avec l'initialiseur de noyau
    tf.keras.initializers.VarianceScaling(mode='fan_avg').
    La couche doit incorporer deux paramètres entraînables,
        gamma et beta, initialisés respectivement à 1 et 0.
    Utilise epsilon = 1e-7.

    Returns:
        Un tenseur correspondant à la sortie activée de la couche.
    """
    layer = tf.keras.layers.Dense(
        n, kernel_initializer=tf.keras.initializers.VarianceScaling(
            mode='fan_avg'))
    Z = layer(prev)
    # Calcul la moyenne et la variance des valeurs dans le batch
    mean, variance = tf.nn.moments(Z, axes=[0])
    # Créer un offset (beta) (initialisé à 0) de taille n entrainable
    offset = tf.Variable(tf.zeros([n]), trainable=True)
    # Créer un scale (gamma) (initialisé à 1) de taille n entrainable
    scale = tf.Variable(tf.ones([n]), trainable=True)
    norm = tf.nn.batch_normalization(
        x=Z,
        mean=mean,
        variance=variance,
        offset=offset,
        scale=scale,
        variance_epsilon=1e-7
    )
    # Active la sortie de la couche
    A = activation(norm)
    return A
