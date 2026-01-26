#!/usr/bin/env python3
"""Bag of words."""
import string
import re
import numpy as np


def bag_of_words(sentences, vocab=None):
    """Create a bag of words embedding matrix based on a list of sentences."""
    features = set()
    words = []
    embeddings = []

    for sentence in sentences:
        text = sentence.lower()
        text = re.findall(r'\b[a-z]{2,}\b', text)
        words.append(text)

    if vocab is not None:
        features = sorted(vocab)
    else:
        features = set()
        for sentence in words:
            features.update(sentence)
        features = sorted(list(features))

    for mots in words:
        embed = []
        for word in sorted(features):
            count = mots.count(word)
            embed.append(count)
        embeddings.append(embed)

    features = np.array(sorted(list(features)))
    return np.array(embeddings), features
