#!/usr/bin/env python3
"""Flip une image horizontalement."""

import tensorflow as tf


def flip_image(image):
    """Flip une image horizontalement."""
    return tf.image.flip_left_right(image)
