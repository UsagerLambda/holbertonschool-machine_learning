#!/usr/bin/env python3
"""Construction de l'architecture ResNet-50."""

from tensorflow import keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
    Construit l'architecture ResNet-50 telle que décrite dans l'article.

    "Deep Residual Learning for Image Recognition" (2015).

    - Les données d'entrée doivent avoir une forme de (224, 224, 3).
    - Toutes les convolutions, à l'intérieur et à l'extérieur des blocs,
      doivent être suivies d'une normalisation par lots (batch normalization)
      sur l'axe des canaux, ainsi que d'une activation ReLU.
    - Toutes les poids doivent être initialisés avec l'initialiseur He normal.
    - La graine (seed) pour l'initialiseur He normal est fixée à zéro.

    Returns:
        keras.Model : le modèle ResNet-50 construit.
    """
    X = K.Input(shape=(224, 224, 3))
    he_init = K.initializers.HeNormal(seed=0)

    conv_1 = K.layers.Conv2D(
        filters=64,
        kernel_size=(7, 7),
        strides=(2, 2),
        activation="relu",
        padding="same",
        kernel_initializer=he_init
    )(X)

    norm_1 = K.layers.BatchNormalization(axis=3)(conv_1)
    activation_1 = K.layers.Activation('relu')(norm_1)
    max_pool_1 = K.layers.MaxPooling2D(pool_size=(3, 3), strides=(
        2, 2), padding="same")(activation_1)

    # Conv2_x: x3 blocs -------------------------------------------------------

    # Ajuste le format d'entrée
    conv2 = projection_block(max_pool_1, [64, 64, 256], s=1)
    conv2 = identity_block(conv2, [64, 64, 256])
    conv2 = identity_block(conv2, [64, 64, 256])

    # Conv3_x: x4 blocs -------------------------------------------------------

    # Ajuste le format d'entrée
    conv3 = projection_block(conv2, [128, 128, 512])
    conv3 = identity_block(conv3, [128, 128, 512])
    conv3 = identity_block(conv3, [128, 128, 512])
    conv3 = identity_block(conv3, [128, 128, 512])

    # Conv4_x: x6 blocs -------------------------------------------------------

    # Ajuste le format d'entrée
    conv4 = projection_block(conv3, [256, 256, 1024])
    conv4 = identity_block(conv4, [256, 256, 1024])
    conv4 = identity_block(conv4, [256, 256, 1024])
    conv4 = identity_block(conv4, [256, 256, 1024])
    conv4 = identity_block(conv4, [256, 256, 1024])
    conv4 = identity_block(conv4, [256, 256, 1024])

    # Conv5_x: x3 blocs -------------------------------------------------------

    # Ajuste le format d'entrée
    conv5 = projection_block(conv4, [512, 512, 2048])
    conv5 = identity_block(conv5, [512, 512, 2048])
    conv5 = identity_block(conv5, [512, 512, 2048])

    # -------------------------------------------------------------------------
    avg_pool = K.layers.AveragePooling2D(pool_size=(
        7, 7), strides=(1, 1))(conv5)
    last_layer = K.layers.Dense(units=1000, activation="softmax",
                                kernel_initializer=he_init)(avg_pool)

    model = K.models.Model(inputs=X, outputs=last_layer)

    return model
