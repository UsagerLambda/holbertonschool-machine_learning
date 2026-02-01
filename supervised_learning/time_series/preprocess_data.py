"""Module de prétraitement des données Bitcoin pour l'entraînement."""

import os
from pathlib import Path

import pandas as pd

# Récupère les fichiers .csv dans le dossier data/
files = list(Path("data").rglob("*.csv"))


def dense_market(df, threshold=0.95):
    """Retourne le premier jour ayant une couverture suffisante en données."""
    # Extrait le jour et la minute de chaque timestamp
    # pour calculer la densité
    df_temp = df.copy()
    df_temp["day"] = df_temp["Timestamp"].dt.date
    df_temp["minute"] = df_temp["Timestamp"].dt.floor("min")

    # Compte le nombre de minutes uniques par jour
    minutes_per_day = df_temp.groupby("day")["minute"].nunique()
    # Calcule le ratio de couverture
    # (minutes présentes / 1440 minutes par jour)
    ratio = minutes_per_day / 1440
    # Filtre les jours atteignant le seuil de couverture
    dense_days = ratio[ratio >= threshold]

    # Si aucun jour n'atteint le ratio
    if dense_days.empty:
        return None

    # Retourne le timestamp du premier jour ayant un ratio conforme
    return pd.Timestamp(dense_days.index[0])


def min_to_hours(df):
    """Agrège les données de la minute vers l'heure."""
    # Rééchantillonne de minute à heure : Close=last, Volumes=sum
    return (
        df.resample("1h", on="Timestamp")
        .agg(
            {
                "Close": "last",
                "Volume_(BTC)": "sum",
                "Volume_(Currency)": "sum",
            }
        )
        .reset_index()
    )


def preprocess(files: list):
    """Nettoie et prépare les fichiers CSV pour l'entraînement."""
    # Pour chaque fichier dans la liste des fichiers
    for file in files:
        print(file)
        df = pd.read_csv(file)  # Lis le fichier avec pandas
        df_new = df.copy()  # Copie le dataframe dans une nouvelle variable
        # (pour ne pas modifier le fichier original)

        # Supprime toutes les lignes qui ont moins de 8 valeurs non nulles
        df_new.dropna(thresh=8, inplace=True)
        # Convertie les valeur de la colonne Timestamp en datetime
        df_new["Timestamp"] = pd.to_datetime(df_new["Timestamp"], unit="s")
        # Supprime les colonnes spécifiées dans la liste
        df_new = df_new.drop(columns=["Open", "High", "Low", "Weighted_Price"])

        # Trouve la première journée ayant au moins 95% de points
        # (1368/1440 minutes)
        first_dense = dense_market(df_new, 0.95)
        # Supprime toutes les données précédent cette date
        df_new = df_new[df_new["Timestamp"] >= first_dense]
        # Rééchantillonne à la minute et remplit les valeurs manquantes
        # par propagation avant (ffill)
        df_new = (
            df_new.set_index("Timestamp")
            .resample("1min")
            .ffill()
            .reset_index()
        )
        # Agrège les données par heure (dernier Close, somme des volumes)
        df_new = min_to_hours(df_new)

        base_name = os.path.basename(file)

        if "bitstamp" in base_name.lower():
            nom_du_fichier = "bitstamp_dataset.csv"
        elif "coinbase" in base_name.lower():
            nom_du_fichier = "coinbase_dataset.csv"
        else:
            raise ValueError("Nom de fichier non reconnu")

        df_new.to_csv(f"dataset/{nom_du_fichier}", index=False)


preprocess(files)
