#!/usr/bin/env python3
"""Insert un document."""


def insert_school(mongo_collection, **kwargs):
    """Insert un document."""
    return mongo_collection.insert_one(kwargs)
