#!/usr/bin/env python3
"""Initialize Keras model using a Adam optimization."""

import tensorflow.keras as K


def optimize_model(network, alpha, beta1, beta2):
    """Set up Adam optimization for a keras model.

    Args:
        network (keras.models): model to optimize
        alpha (float): learning rate
        beta1 (float): first Adam optimization parameter
        beta2 (float): seconde Adam optimization parameter

    Returns:
        None
    """
    optimizer = K.optimizers.Adam(
        learning_rate=alpha,
        beta_1=beta1,
        beta_2=beta2
    )
    network.compile(optimizer=optimizer,
                    loss='categorical_crossentropy',
                    metrics=['accuracy'])
    return None
