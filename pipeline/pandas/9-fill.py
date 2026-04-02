#!/usr/bin/env python3

"""Remove Weighted_Price, fill les colonnes vide."""


def fill(df):
    """Remove Weighted_Price, fill les colonnes vide."""
    df = df.drop(columns="Weighted_Price").copy()
    df["Close"] = df["Close"].ffill()
    for col in ["High", "Low", "Open"]:
        df[col] = df[col].fillna(df['Close'])
    df[
        ["Volume_(BTC)", "Volume_(Currency)"]] = df[
            ["Volume_(BTC)", "Volume_(Currency)"]].fillna(0)
    return df
