#!/usr/bin/env python3

from tensorflow import keras as K
import ssl
import time
ssl._create_default_https_context = ssl._create_unverified_context
from plot import plot_training_history, print_training_summary
from config import LEARNING_RATE, BATCH_SIZE, EPOCHS, DROPOUT_RATE, DENSE_UNITS, MODEL_NAME, PATIENCE, TRAINABLE, NB_UNFREEZE_LAYERS, PLOT

# =========================================================================== #

def preprocess_data(X, Y):
    """Prétraite les données pour un modèle choisis.

    Args:
        X (numpy.ndarray): (m, 32, 32, 3) contenant les images CIFAR-10, où m est le nombre d'exemples.
        Y (numpy.ndarray): (m,) contenant les étiquettes CIFAR-10 correspondant à X.

    Returns:
        X_p (numpy.ndarray): données X prétraitées.
        Y_p (numpy.ndarray): labels Y prétraités.
    """
    X = X.astype('float32')

    # Prétraitement spécifique (normalisation uniquement)
    if MODEL_NAME == 'MobileNetV2':
        X_p = K.applications.mobilenet_v2.preprocess_input(X)
    elif MODEL_NAME == 'ResNet50':
        X_p = K.applications.resnet50.preprocess_input(X)
    elif MODEL_NAME == 'EfficientNetV2B1':
        X_p = K.applications.efficientnet_v2.preprocess_input(X)

    # Conversion des labels en format catégoriel
    Y_p = K.utils.to_categorical(Y, 10)

    return X_p, Y_p

# =========================================================================== #

if __name__ == "__main__":
    print(f"🚀 Début de l'entraînement avec {MODEL_NAME}...")
    start_time = time.time()

    # Chargement des données CIFAR-10
    (x_train, y_train), (x_test, y_test) = K.datasets.cifar10.load_data()

    # Prétraitement des données (SANS redimensionnement)
    X_p, Y_p = preprocess_data(x_train, y_train)
    X_val, Y_val = preprocess_data(x_test, y_test)

    # Augmentation de données
    train_datagen = K.preprocessing.image.ImageDataGenerator(
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        horizontal_flip=True,
        zoom_range=0.2
    )
    train_gen = train_datagen.flow(X_p, Y_p, batch_size=BATCH_SIZE)

    # Récupère le model en fonction de la variable
    if MODEL_NAME == 'MobileNetV2':
        base_model_func = K.applications.MobileNetV2
        target_size = 224
    elif MODEL_NAME == 'ResNet50':
        base_model_func = K.applications.ResNet50
        target_size = 224
    elif MODEL_NAME == 'EfficientNetV2B1':
        base_model_func = K.applications.EfficientNetV2B1
        target_size = 224

    inputs = K.Input(shape=(32, 32, 3))

    # Augmentation directement dans le modèle
    x = K.layers.RandomFlip("horizontal")(inputs)
    x = K.layers.RandomRotation(0.05)(x)
    x = K.layers.RandomZoom(0.1)(x)

    # Resize des données ici
    x = K.layers.Resizing(target_size, target_size)(x)

    # Base model avec input_tensor
    base_model = base_model_func(
        include_top=False,
        weights='imagenet',
        input_tensor=x,
        pooling='avg'
    )

    # Gel des couches
    base_model.trainable = False

    if TRAINABLE == True:  # Dégèle de certaine couches si TRAINABLE est True
        for layer in base_model.layers[-NB_UNFREEZE_LAYERS:]:
            layer.trainable = True

    # Tête de classification
    x = K.layers.Dense(DENSE_UNITS, activation='relu')(base_model.output)
    x = K.layers.Dropout(DROPOUT_RATE)(x)
    outputs = K.layers.Dense(10, activation='softmax')(x)

    model = K.Model(inputs, outputs)

    model.compile(
        optimizer=K.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    training_start = time.time()

    callbacks = [
        # Sauvegarde le meilleur modèle (selon val_accuracy)
        K.callbacks.ModelCheckpoint('cifar10.h5', save_best_only=True, monitor='val_accuracy'),
        # Arrête l'entraînement si la validation n'évolue plus (patience configurable)
        K.callbacks.EarlyStopping(patience=PATIENCE, monitor='val_accuracy'),
        # Réduit le learning rate si la validation stagne
        K.callbacks.ReduceLROnPlateau(factor=0.5, patience=3, monitor='val_accuracy')
    ]

    # Entraînement du modèle principal
    history = model.fit(
        X_p, Y_p,  # Données d'entraînement
        validation_data=(X_val, Y_val),  # Données de validation pour le monitoring
        batch_size=BATCH_SIZE,  # Taille des mini-batchs
        epochs=EPOCHS,  # Nombre d'époques
        shuffle=True,  # Mélange les données à chaque époque
        callbacks=callbacks,  # Liste des callbacks pour l'optimisation
        verbose=1  # Affiche la progression détaillée
    )

    training_end = time.time()
    end_time = time.time()

    total_time = end_time - start_time
    training_time = training_end - training_start

    if PLOT:
        print_training_summary(history, training_time, total_time)
    plot_training_history(history)

    print("✅ Entraînement terminé!")
