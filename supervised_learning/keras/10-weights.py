#!/usr/bin/env python3
"""Save and Load models."""

import tensorflow.keras as K


def save_weights(network, filename, save_format='keras'):
    """Save the weights of the model.

    Args:
        network (model.keras): model from were we save weights
        filename (string): name of the file
        save_format (str, optional): extension format of the file (depreciated)
        Defaults to 'keras'.

    Returns:
        None: none
    """
    network.save_weights(filename, save_format=save_format)
    return None


def load_weights(network, filename):
    """Load weights.

    Args:
        network (model.keras): model weights to load
        filename (string): name of the file that contain the saved weights

    Returns:
        None: none
    """
    network.load_weights(filename)
    return None
