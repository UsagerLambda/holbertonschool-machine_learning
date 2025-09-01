#!/usr/bin/env python3
"""Save and Load models."""

import tensorflow.keras as K


def save_config(network, filename):
    """Save a model configurations in JSON format.

    Args:
        network (keras.Model): he model whose configuration should be save
        filename (str): the path of the file that gonna save the configuration

    Returns:
        None: none
    """
    config = network.to_json()
    with open(filename, 'w') as f:
        f.write(config)
    return None


def load_config(filename):
    """Load a model with specific configuration.

    Args:
        filename (str): the path of the file

    Returns:
        keras.Model: the loaded model
    """
    with open(filename, 'r') as f:
        config = f.read()
    model = K.models.model_from_json(config)
    return model
