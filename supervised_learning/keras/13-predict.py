#!/usr/bin/env python3
"""Make prediction using neural network."""

import tensorflow.keras as K


def predict(network, data, verbose=False):
    """Make prediction using a neural network.

    Args:
        network (keras.Model): model to predict
        data (numpy.ndarray): input data to make prediction with
        verbose (bool, optional): Determine if output should
            be printed during the training process. Defaults to True.

    Returns:
        numpy.ndarray: array of prediction
    """
    prediction = network.predict(data, verbose=verbose)
    return prediction
