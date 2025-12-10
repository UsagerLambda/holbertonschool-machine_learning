#!/usr/bin/env python3

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

import warnings

warnings.filterwarnings("ignore")

import time

import GPyOpt
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import datasets, layers, models, regularizers


def load_cifar():
    (train_images, train_labels), (test_images, test_labels) = (
        datasets.cifar10.load_data()
    )

    train_images, test_images = train_images / 255.0, test_images / 255.0

    train_labels = tf.keras.utils.to_categorical(train_labels, 10)
    test_labels = tf.keras.utils.to_categorical(test_labels, 10)

    X_train, X_val, Y_train, Y_val = train_test_split(
        train_images, train_labels, test_size=0.2, random_state=42
    )

    return X_train, Y_train, X_val, Y_val, test_images, test_labels


def build_model(dense_units, dropout, l2_reg, hidden_layers, learning_rate):
    """
    Construit et compile un CNN pour CIFAR-10

    :param dense_units: nombre de neurones dans les couches Dense
    :param dropout: taux de dropout
    :param l2_reg: coefficient de régularisation L2
    :param hidden_layers: nombre de couches Dense cachées
    :param learning_rate: learning rate pour l'optimizer
    """
    model = models.Sequential()

    model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 3)))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(64, (3, 3), activation="relu"))
    model.add(layers.MaxPooling2D((2, 2)))
    model.add(layers.Dropout(0.2))

    model.add(layers.Conv2D(128, (3, 3), activation="relu"))
    model.add(layers.Dropout(0.2))

    model.add(layers.Flatten())

    for _ in range(hidden_layers):
        model.add(
            layers.Dense(
                dense_units,
                activation="relu",
                kernel_regularizer=regularizers.l2(l2_reg),
            )
        )
        model.add(layers.Dropout(dropout))

    model.add(layers.Dense(10))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )

    return model


def objective_function(hyperparams):
    """
    Fonction objectif pour GPyOpt

    Args:
        hyperparams (np.ndarray): array de forme (1, 6)
            contenant les hyperparamètres.
    """
    start_time = time.time()
    global X_train, Y_train, X_val, Y_val

    batch_size = int(hyperparams[0, 0])
    learning_rate = float(hyperparams[0, 1])
    dropout = float(hyperparams[0, 2])
    dense_units = int(hyperparams[0, 3])
    l2_reg = float(hyperparams[0, 4])
    hidden_layers = int(hyperparams[0, 5])

    print(
        f"\n🔧 Testing: bs={batch_size}, lr={learning_rate:.5f}, dropout={dropout:.2f}, "
        f"units={dense_units}, l2={l2_reg:.5f}, layers={hidden_layers}"
    )

    model = build_model(
        dense_units=dense_units,
        dropout=dropout,
        l2_reg=l2_reg,
        hidden_layers=hidden_layers,
        learning_rate=learning_rate,
    )

    filename = f"model_bs{batch_size}_lr{learning_rate:.5f}_drop{dropout:.2f}_units{dense_units}_l2_{l2_reg:.5f}_layers{hidden_layers}.keras"

    # Sauvegarde le meilleur modèle
    checkpoint = tf.keras.callbacks.ModelCheckpoint(
        filepath=filename,
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=0,
    )

    # Stop l'entrainement
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor="val_loss", patience=5, restore_best_weights=True, verbose=0
    )

    # Entrainement du modèle
    history = model.fit(
        X_train,
        Y_train,  # Données d'entraînement
        batch_size=batch_size,  # Taille des mini-batchs
        epochs=50,  # Nombre d'époques
        validation_data=(X_val, Y_val),
        callbacks=[checkpoint, early_stop],
        verbose=0,
    )

    Y_pred = model.predict(X_val, verbose=0)
    Y_pred_classes = tf.argmax(Y_pred, axis=1).numpy()
    Y_val_classes = tf.argmax(Y_val, axis=1).numpy()

    f1 = f1_score(Y_val_classes, Y_pred_classes, average="macro")

    elapsed_time = time.time() - start_time

    print(f"✅ F1-score: {f1:.4f} | Time: {elapsed_time:.1f}s")

    return 1 - f1


if __name__ == "__main__":
    X_train, Y_train, X_val, Y_val, X_test, Y_test = load_cifar()

    print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

    bounds = [
        {"name": "batch_size", "type": "discrete", "domain": (32, 64, 128)},
        {"name": "learning_rate", "type": "continuous", "domain": (0.0001, 0.003)},
        {"name": "dropout", "type": "continuous", "domain": (0.1, 0.4)},
        {"name": "dense_units", "type": "discrete", "domain": (128, 256)},
        {"name": "l2_reg", "type": "continuous", "domain": (0.0001, 0.005)},
        {"name": "hidden_layers", "type": "discrete", "domain": (1, 2)},
    ]

    optimizer = GPyOpt.methods.BayesianOptimization(
        f=objective_function,
        domain=bounds,
        model_type="GP",
        acquisition_type="EI",
        acquisition_jitter=0.05,
        exact_feval=True,
        maximize=False,
    )

    print("\n🚀 Début de l'optimisation bayésienne...")

    total_start = time.time()

    optimizer.run_optimization(max_iter=25)

    total_time = time.time() - total_start

    print(f"\n✅ Optimisation terminée en {total_time / 60:.1f} minutes")

    os.makedirs("img", exist_ok=True)
    optimizer.plot_convergence()
    plt.savefig("img/convergence_plot.png", dpi=150, bbox_inches="tight")
    print("✅ Plot sauvegardé : img/convergence_plot.png")

    best_params = optimizer.x_opt
    best_f1 = 1 - optimizer.fx_opt

    with open("bayes_opt.txt", "w") as f:
        f.write("=" * 60 + "\n")
        f.write("BAYESIAN OPTIMIZATION REPORT - CIFAR-10 CNN\n")
        f.write("=" * 60 + "\n\n")

        f.write(f"Total optimization time: {total_time / 60:.2f} minutes\n")
        f.write(f"Number of iterations: {len(optimizer.Y)}\n\n")

        f.write("BEST HYPERPARAMETERS FOUND:\n")
        f.write("-" * 60 + "\n")
        f.write(f"  Batch size:         {int(best_params[0])}\n")
        f.write(f"  Learning rate:      {best_params[1]:.6f}\n")
        f.write(f"  Dropout:            {best_params[2]:.4f}\n")
        f.write(f"  Dense units:        {int(best_params[3])}\n")
        f.write(f"  L2 regularization:  {best_params[4]:.6f}\n")
        f.write(f"  Hidden layers:      {int(best_params[5])}\n\n")

        f.write(f"BEST F1-SCORE: {best_f1:.4f}\n\n")

        f.write("ALL EVALUATIONS:\n")
        f.write("=" * 60 + "\n")
        for i, (X, Y) in enumerate(zip(optimizer.X, optimizer.Y)):
            f1_iter = 1 - Y[0]
            f.write(f"Iter {i + 1:2d} | F1: {f1_iter:.4f} | ")
            f.write(f"Iter {i + 1:2d} | F1: {f1_iter:.4f} | ")
            f.write(f"bs={int(X[0]):3d} lr={X[1]:.5f} drop={X[2]:.2f} ")
            f.write(f"units={int(X[3]):3d} l2={X[4]:.5f} layers={int(X[5])}\n")
