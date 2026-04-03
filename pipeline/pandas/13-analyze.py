#!/usr/bin/env python3

"""Descriptive statistics."""


def analyze(df):
    """Descriptive statistics."""
    df = df.drop(columns=['Timestamp'])
    return df.describe()
