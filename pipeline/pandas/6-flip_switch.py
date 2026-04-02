#!/usr/bin/env python3

"""Reverse et transpose le Dataframe."""


def flip_switch(df):
    """Reverse et transpose le Dataframe."""
    return df.T.iloc[:, ::-1]
