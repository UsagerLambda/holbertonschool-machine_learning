#!/usr/bin/env python3

"""Sort Dataframe by High descending order."""


def high(df):
    """Sort Dataframe by High descending order."""
    return df.sort_values(by="High", ascending=False)
