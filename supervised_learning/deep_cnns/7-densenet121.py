#!/usr/bin/env python3


from tensorflow import keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    X = K.Input(shape=(224, 224, 3))
    he_init = K.initializers.HeNormal(seed=0)

    # Initial number of filters
    nb_filters = 64

    # Initial convolution: 7x7 conv, stride 2
    bn = K.layers.BatchNormalization(axis=3)(X)
    relu = K.layers.Activation('relu')(bn)
    conv = K.layers.Conv2D(
        filters=nb_filters,
        kernel_size=(7, 7),
        strides=(2, 2),
        padding="same",
        kernel_initializer=he_init
    )(relu)

    # Max pooling: 3x3, stride 2
    max_pool = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding="same"
    )(conv)

    # Dense Block (1): 6 layers
    dense, nb_filters = dense_block(max_pool, nb_filters, growth_rate, 6)

    # Transition Layer (1)
    trans, nb_filters = transition_layer(dense, nb_filters, compression)

    # Dense Block (2): 12 layers
    dense, nb_filters = dense_block(trans, nb_filters, growth_rate, 12)

    # Transition Layer (2)
    trans, nb_filters = transition_layer(dense, nb_filters, compression)

    # Dense Block (3): 24 layers
    dense, nb_filters = dense_block(trans, nb_filters, growth_rate, 24)

    # Transition Layer (3)
    trans, nb_filters = transition_layer(dense, nb_filters, compression)

    # Dense Block (4): 16 layers
    dense, nb_filters = dense_block(trans, nb_filters, growth_rate, 16)

    # Global Average Pooling: 7x7
    avg_pool = K.layers.AveragePooling2D(
        pool_size=(7, 7),
        strides=(1, 1)
    )(dense)

    # Classification Layer: 1000D fully-connected, softmax
    last_layer = K.layers.Dense(
        units=1000,
        activation="softmax",
        kernel_initializer=he_init
    )(avg_pool)

    model = K.models.Model(inputs=X, outputs=last_layer)

    return model
