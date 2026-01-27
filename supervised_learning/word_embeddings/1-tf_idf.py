#!/usr/bin/env python3
"""TF-IDF implementation."""
import re
import numpy as np


def tf_idf(sentences, vocab=None):
    """Create a TF-IDF embedding matrix based on a list of sentences."""
    features = set()
    words = []
    embeddings = []

    for sentence in sentences:
        text = sentence.lower()
        text = re.findall(r'\b[a-z]{2,}\b', text)
        words.append(text)

    if vocab is not None:
        features = list(vocab)
    else:
        features = set()
        for sentence in words:
            features.update(sentence)
        features = sorted(list(features))

    corpusSize = len(sentences)

    idf = []
    for word in features:
        doc_count = sum(1 for doc in words if word in doc)
        if doc_count > 0:
            idf_value = np.log((1 + corpusSize) / (1 + doc_count)) + 1
        else:
            idf_value = 0
        idf.append(idf_value)

    embeddings = []

    for mots in words:

        tf = []
        for word in features:
            if len(mots) > 0:
                tf_value = mots.count(word) / len(mots)
            else:
                tf_value = 0
            tf.append(tf_value)

        tfidf = [tf[i] * idf[i] for i in range(len(features))]
        norm = np.sqrt(sum(x**2 for x in tfidf))
        if norm > 0:
            tfidf = [x / norm for x in tfidf]
        embeddings.append(tfidf)

    return np.array(embeddings), np.array(features)
