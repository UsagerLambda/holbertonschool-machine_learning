#!/usr/bin/env python3
"""Change la teinte d'une image."""

import tensorflow as tf


def change_hue(image, delta):
    """Change la teinte d'une image."""
    return tf.image.adjust_hue(image, delta)
