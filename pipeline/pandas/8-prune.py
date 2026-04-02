#!/usr/bin/env python3

"""Drop row when Close is NaN."""


def prune(df):
    """Drop row when Close is NaN."""
    return df.dropna(subset=['Close'])
