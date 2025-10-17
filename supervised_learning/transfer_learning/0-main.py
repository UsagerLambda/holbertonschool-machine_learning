#!/usr/bin/env python3

from tensorflow import keras as K
preprocess_data = __import__('0-transfer').preprocess_data


# to fix issue with saving keras applications
if not hasattr(K.backend, 'learning_phase'):
    K.backend.learning_phase = lambda: False

_, (X, Y) = K.datasets.cifar10.load_data()
X_p, Y_p = preprocess_data(X, Y)
model = K.models.load_model('cifar10.h5', safe_mode=False)
model.evaluate(X_p, Y_p)
