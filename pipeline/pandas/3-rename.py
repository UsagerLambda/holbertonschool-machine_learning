#!/usr/bin/env python3

"""Update Datetime."""

import pandas as pd


def rename(df):
    """Change le nom de la colonne Timestamp par Datetime.

    Change le format de date.
    Drop les colonnes que nous ne voulons pas afficher.
    """
    df = df.rename(columns={"Timestamp": "Datetime"})
    df["Datetime"] = pd.to_datetime(df["Datetime"], unit='s')
    df = df[["Datetime", "Close"]]
    return df
