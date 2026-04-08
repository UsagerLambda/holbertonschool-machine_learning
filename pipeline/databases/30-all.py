#!/usr/bin/env python3
"""Liste tous les documents d'une collection."""


def list_all(mongo_collection):
    """List tous les documents d'une collection."""
    return mongo_collection.find({})
