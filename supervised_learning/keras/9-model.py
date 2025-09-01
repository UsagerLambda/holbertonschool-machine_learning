#!/usr/bin/env python3
"""Save and Load models."""

import tensorflow.keras as K


def save_model(network, filename):
    """Save model.

    Args:
        network (model.keras): model to save
        filename (string): name and path of the file

    Returns:
        None: none
    """
    network.save(filename)
    return None


def load_model(filename):
    """Load model.

    Args:
        filename (string): name of the file to load

    Returns:
        model.keras: model loaded
    """
    model = K.models.load_model(filename)
    return model
