#!/usr/bin/env python3
"""Chnage le contraste d'une image."""

import tensorflow as tf


def change_contrast(image, lower, upper):
    """Chnage le contraste d'une image."""
    return tf.image.random_contrast(
        image,
        lower,
        upper
    )
