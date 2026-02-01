#!/usr/bin/env python3
"""Module de prédiction du prix Bitcoin avec un réseau LSTM."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Chargement des datasets process par preprocess_data.py
train = pd.read_csv("dataset/bitstamp_dataset.csv")
test = pd.read_csv("dataset/coinbase_dataset.csv")

WINDOW_SIZE = 24
EPOCHS = 100
LEARNING_RATE = 0.001
BATCH_SIZE = 32
DROPOUT = 0.2
EARLY_STOPPING = 20
CLIPNORM = 1.0


def standardise_train(df_train):
    """
    Standardise les colonnes et crée des colonnes suffixées _std.

    Retourne les moyennes et écarts-types pour réutilisation sur le test.
    """
    columns = ["Close", "Volume_(BTC)", "Volume_(Currency)"]
    means = {}
    stds = {}
    for col in columns:
        means[col] = df_train[col].mean()
        stds[col] = df_train[col].std()
        df_train[f"{col}_std"] = (df_train[col] - means[col]) / stds[col]
    return means, stds, df_train


def standardise_test(df_test, means, stds):
    """Standardise les colonnes du test avec les paramètres du train."""
    columns = ["Close", "Volume_(BTC)", "Volume_(Currency)"]
    for col in columns:
        df_test[f"{col}_std"] = (df_test[col] - means[col]) / stds[col]
    return df_test


# Appel pour la standarisation
means, stds, train = standardise_train(train)
test = standardise_test(test, means, stds)


def data_to_input_and_output(df):
    """Convertit un DataFrame en données d'entrée et de sortie."""
    input_data = []
    output_data = []
    # Colonnes normalisées pour l'input
    feature_cols = ["Close_std", "Volume_(BTC)_std", "Volume_(Currency)_std"]
    # Colonne Close, non normalisée pour pouvoir calculer le changement
    close_values = df["Close"].values

    # Itère de 0 jusqu'à l'index max
    # permettant une fenêtre complète + une cible
    for index in range(len(df) - WINDOW_SIZE):
        # Récupère une fenêtre de WINDOW_SIZE lignes à partir de l'index
        input_sample = (
            df[feature_cols].iloc[index:index + WINDOW_SIZE].values
        )

        # Prix à la dernière heure de la fenêtre
        # (index 23 = 24ème heure, car index 0 = 1ère heure)
        current_close = close_values[index + WINDOW_SIZE - 1]
        # Prix à l'heure suivante (à prédire)
        next_close = close_values[index + WINDOW_SIZE]
        # Différence entre le prix à l'heure 23 et la 24
        change = next_close - current_close

        input_data.append(input_sample)  # Ajoute la fenêtre de 24h
        output_data.append(
            change
        )  # Ajoute le changement correspondant (en USD)

    return np.array(input_data), np.array(output_data)


# Fait les appels pour créer les input et les output
# pour les données d'entrainement et celle de test
train_input, train_output_raw = data_to_input_and_output(train)
test_input, test_output_raw = data_to_input_and_output(test)

# Standarisation des output (différence entre le prix à l'heure 23 et la 24)
output_mean = train_output_raw.mean()
output_std = train_output_raw.std()

# Standardisation du train,
# puis standardisation de test grace aux mean et std de train
train_output = (train_output_raw - output_mean) / output_std
test_output = (test_output_raw - output_mean) / output_std

print(f"Output normalization: mean={output_mean:.2f}, std={output_std:.2f}")

# Sauvegarder les paramètres de standardisation
np.savez(
    "normal/normalization_params.npz",
    means=means,
    stds=stds,
    output_mean=output_mean,
    output_std=output_std,
)


# Calcule la taille du train (80%)
train_size = int(0.8 * len(train_input))

# Créer le dataset d'entraînement par rapport à la taille définie (train_size)
# Prend les 80 premiers % de train_input & train_output
# (entrée et sortie attendue)
train_ds = tf.data.Dataset.from_tensor_slices(
    (train_input[:train_size], train_output[:train_size])
)
# Mélange les données, crée des batch de 32, et précharge les données
# Shuffle évite au modèle d'apprendre l'ordre chronologique
# au lieu des vrais patterns
train_ds = train_ds.shuffle(10000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Même chose mais avec les 20% restants pour créer un dataset de validation
# Permet d'évaluer le modèle sur des données
# qu'il n'a pas vues pendant l'entraînement
val_ds = tf.data.Dataset.from_tensor_slices(
    (train_input[train_size:], train_output[train_size:])
)
# Pas de shuffle car ce dataset sert uniquement à évaluer (pas à entraîner)
val_ds = val_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Créer le dataset de test avec toutes les données de coinbase
test_ds = tf.data.Dataset.from_tensor_slices((test_input, test_output))
test_ds = test_ds.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)


# ========== MODÈLE ==========
def train_neural_network(
    train_ds, val_ds, epochs=EPOCHS, learning_rate=LEARNING_RATE
):
    """Crée et entraîne un réseau LSTM pour la prédiction de prix."""
    model = tf.keras.models.Sequential(
        [
            tf.keras.layers.InputLayer((WINDOW_SIZE, 3)),
            tf.keras.layers.LSTM(64, return_sequences=True),
            tf.keras.layers.Dropout(DROPOUT),
            tf.keras.layers.LSTM(64),
            tf.keras.layers.Dropout(DROPOUT),
            tf.keras.layers.Dense(8, "relu"),
            tf.keras.layers.Dropout(DROPOUT),
            tf.keras.layers.Dense(1, "linear"),
        ]
    )

    model.compile(
        loss="mse",
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=learning_rate, clipnorm=CLIPNORM
        ),
        metrics=["mae", "mse"],
    )

    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            "model/best.keras", save_best_only=True, monitor="val_loss"
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=EARLY_STOPPING,
            restore_best_weights=True,
        ),
    ]

    model.fit(
        train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks
    )
    return model


# Entrainement
model = train_neural_network(train_ds, val_ds)

# Utilisation du model entrainé sur toutes les données de bitstamp & coinbase
# (pas d'entrainement)
train_pred_norm = model.predict(
    train_input
).flatten()  # Performance sur données d'entrainement
test_pred_norm = model.predict(
    test_input
).flatten()  # Performance sur données de test

# Dé-normaliser les predictions pour retourner au valeur USD
train_pred = train_pred_norm * output_std + output_mean
test_pred = test_pred_norm * output_std + output_mean


def reconstruct_prices(df, predictions, actual_changes):
    """
    Reconstruit les prix absolus à partir des variations prédites et réelles.

    df: DataFrame contenant la colonne 'Close' avec les prix historiques.
    predictions: Array des variations de prix prédites par le modèle (en USD).
    actual_changes: Array des variations de prix réelles (en USD).
    """
    close_values = df["Close"].values
    pred_prices = []
    real_prices = []

    for i in range(len(predictions)):
        # Prix actuel
        current_price = close_values[i + WINDOW_SIZE - 1]
        # Prix prédit = Prix actuel + Changement prédit
        pred_prices.append(current_price + predictions[i])
        # Prix réel = Prix actuel + Changement réel
        real_prices.append(current_price + actual_changes[i])

    return np.array(pred_prices), np.array(real_prices)


# Appel à reconstruct_prices,
# récupère les valeurs prédites et les valeurs réelles
train_pred_prices, train_real_prices = reconstruct_prices(
    train, train_pred, train_output_raw
)
test_pred_prices, test_real_prices = reconstruct_prices(
    test, test_pred, test_output_raw
)

plt.figure(figsize=(14, 5))
plt.plot(train_real_prices[:200], label="Réel")
plt.plot(train_pred_prices[:200], label="Prédit")
plt.legend()
plt.title("Train")
plt.savefig("plots/train.png")

plt.figure(figsize=(14, 5))
plt.plot(test_real_prices[:200], label="Réel")
plt.plot(test_pred_prices[:200], label="Prédit")
plt.legend()
plt.title("Test")
plt.savefig("plots/test.png")
