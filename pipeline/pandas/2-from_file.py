#!/usr/bin/env python3
"""Lis un csv et delimite les données retournée."""

import pandas as pd


def from_file(filename, delimiter):
    """Lis un csv et delimite les données retournée."""
    return pd.read_csv(filename, delimiter=delimiter)
