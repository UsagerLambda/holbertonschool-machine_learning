#!/usr/bin/env python3
"""Module containing L2 regularization gradient descent implementation."""

import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """Create a tensorflow layer with L2 regularization.

    Args:
        prev (keras.Tensor): tensor containing the output of the previous layer
        n (int): number of nodes the new layer should contain
        activation (function): activation function that should be used on the
            layer
        lambtha (float): L2 regularization parameter

    Returns:
        keras.Tensor: output of the new layer
    """
    layer = tf.keras.layers.Dense(
        units=n,
        activation=activation,
        kernel_regularizer=tf.keras.regularizers.L2(lambtha)
    )
    return layer(prev)
