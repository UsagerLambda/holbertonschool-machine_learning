#!/usr/bin/env python3
"""Update les documents {name}."""


def update_topics(mongo_collection, name, topics):
    """Update les documents {name}."""
    return mongo_collection.update_many(
        {"name": name}, {"$set": {"topics": topics}}
        )
