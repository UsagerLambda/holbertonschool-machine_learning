#!/usr/bin/env python3

"""Récupère des valeurs tous les 60 lignes."""


def slice(df):
    """Récupère des valeurs tous les 60 lignes."""
    df = df[["High", "Low", "Close", "Volume_(BTC)"]]
    df = df.iloc[::60, :]
    return df
