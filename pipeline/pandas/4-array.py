#!/usr/bin/env python3

"""Convert les 10 dernier rows High & Close du DF en tableau numpy."""


def array(df):
    """Convert les 10 dernier rows High & Close du DF en tableau numpy."""
    df = df[["High", "Close"]]
    df = df.tail(10)
    df = df.to_numpy()
    return df
