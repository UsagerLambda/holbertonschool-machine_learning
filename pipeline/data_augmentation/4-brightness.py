#!/usr/bin/env python3
"""Change la luminosité d'une image."""

import tensorflow as tf


def change_brightness(image, max_delta):
    """Change la luminosité d'une image."""
    return tf.image.random_brightness(image, max_delta)
