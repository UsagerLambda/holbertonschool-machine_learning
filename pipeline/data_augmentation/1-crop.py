#!/usr/bin/env python3
"""Crop une image."""

import tensorflow as tf


def crop_image(image, size):
    """Crop une image."""
    return tf.image.random_crop(
        image,
        size
    )
