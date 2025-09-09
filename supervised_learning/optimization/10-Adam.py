#!/usr/bin/env python3
"""Set up the Adam optimization algorithm in TensorFlow."""

import numpy as np
import tensorflow as tf


def create_Adam_op(alpha, beta1, beta2, epsilon):
    """Set up the Adam optimization algorithm in TensorFlow.

    Args:
        alpha (float): learning rate
        beta1 (float): weight used for the first moment
        beta2 (float): weight used for the second moment
        epsilon (float): small number to avoid division by zero

    Returns:
        Keras.Adam: tensoflow Adam object
    """
    optimizer = tf.keras.optimizers.Adam(
        learning_rate=alpha,
        beta_1=beta1,
        beta_2=beta2,
        epsilon=epsilon
    )
    return optimizer
