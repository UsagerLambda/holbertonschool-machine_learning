#!/usr/bin/env python3

"""Défini Timestamp comme index du Dataframe."""


def index(df):
    """Défini Timestamp comme index du Dataframe."""
    return df.set_index('Timestamp')
