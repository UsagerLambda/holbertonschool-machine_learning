#!/usr/bin/env python3
"""Renvoie les documents répondant au filtre."""


def schools_by_topic(mongo_collection, topic):
    """Renvoie les documents répondant au filtre."""
    return mongo_collection.find({"topics": topic})
