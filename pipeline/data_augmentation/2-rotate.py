#!/usr/bin/env python3
"""Rotate une image."""

import tensorflow as tf


def rotate_image(image):
    """Rotate une image."""
    return tf.image.rot90(image)
