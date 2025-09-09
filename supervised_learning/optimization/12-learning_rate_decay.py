#!/usr/bin/env python3
"""Create a alpha operation in tensorflow using inverse time decay."""

import tensorflow as tf


def learning_rate_decay(alpha, decay_rate, decay_step):
    """Create a alpha operation in tensorflow using inverse time decay.

    Args:
        alpha (float): original learning rate
        decay_rate (int): weight used to determine the rate
            at which alpha will decay
        decay_step (int): number of passes of gradient descent
            that should occur before alpha is decayed further

    Returns:
        Tensorflow.operation: alpha operation
    """
    op = tf.keras.optimizers.schedules.InverseTimeDecay(
        initial_learning_rate=alpha,
        decay_steps=decay_step,
        decay_rate=decay_rate,
        staircase=True
    )
    return op
