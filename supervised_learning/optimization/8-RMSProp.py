#!/usr/bin/env python3
"""Set up the RMSProp optimization algorithm in TensorFlow."""

import tensorflow as tf


def create_RMSProp_op(alpha, beta2, epsilon):
    """Set up the RMSProp optimization algorithm in TensorFlow.

    Args:
        alpha (float): learning rate
        beta2 (float): RMSProp weight (Discounting factor)
        epsilon (float): small number to avoid division by zero

    Returns:
        ft.Optimizers: Tensorflow RMSprop object
    """
    optimizer = tf.keras.optimizers.RMSprop(
        learning_rate=alpha,
        momentum=beta2,
        epsilon=epsilon
    )
    return optimizer
