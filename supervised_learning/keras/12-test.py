#!/usr/bin/env python3
"""Test a neural network."""

import tensorflow.keras as K


def test_model(network, data, labels, verbose=True):
    """Test a neural network.

    Args:
        network (keras.Model): model to test
        data (numpy.ndarray): input data to test
        labels (numpy.ndarray): correct one-hot labels of data
        verbose (bool, optional): Determine if output should
            be printed during the training process. Defaults to True.

    Returns:
        list: list that contain the loss and accuracy of the model
    """
    score = network.evaluate(data, labels, verbose=verbose)
    return score
