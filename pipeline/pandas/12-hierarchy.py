#!/usr/bin/env python3
"""Module pour la concaténation et la hiérarchisation de DataFrames pandas."""

import pandas as pd
index = __import__('10-index').index


def concat(df1, df2):
    """Concatène df2 (bitstamp) jusqu'au timestamp 1417411920 sur df1."""
    df1 = index(df1)
    df2 = index(df2).loc[:'1417411920']
    return pd.concat([df2, df1], keys=["bitstamp", "coinbase"])


def hierarchy(df1, df2):
    """Filtre les deux DF entre les timestamps 1417411980 et 1417417980.

    Concatène avec des clés hiérarchiques et retourne un index trié.
    """
    df1 = index(df1).loc['1417411980':'1417417980']
    df2 = index(df2).loc['1417411980':'1417417980']
    result = pd.concat([df2, df1], keys=["bitstamp", "coinbase"])
    return result.swaplevel(0, 1).sort_index()
