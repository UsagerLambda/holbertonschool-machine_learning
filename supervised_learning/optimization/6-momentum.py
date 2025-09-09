#!/usr/bin/env python3
"""Sets up gradient descend with momentum using Tensorflow."""

import tensorflow as tf


def create_momentum_op(alpha, beta1):
    """Set up gradient descent with momentum using Tensorflow.

    Args:
        alpha (float): learning rate
        beta1 (float): momentum

    Returns:
        tf.MomentumOptimizer: optimizer object
    """
    optimizer = tf.compat.v1.train.MomentumOptimizer(
        learning_rate=alpha,
        momentum=beta1,
        use_nesterov=False
    )
    return optimizer
