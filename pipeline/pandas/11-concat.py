#!/usr/bin/env python3
"""Concatène deux DataFrames pandas indexés par Timestamp."""

import pandas as pd
index = __import__('10-index').index


def concat(df1, df2):
    """Concatène df2 (bitstamp) jusqu'au timestamp 1417411920 sur df1."""
    df1 = index(df1)
    df2 = index(df2).loc[:'1417411920']
    return pd.concat([df2, df1], keys=["bitstamp", "coinbase"])
