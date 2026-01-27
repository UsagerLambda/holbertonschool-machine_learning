#!/usr/bin/env python3
"""Word to vector."""

from gensim.models import Word2Vec
import os


def word2vec_model(
            sentences,
            vector_size=100,
            min_count=5,
            window=5,
            negative=5,
            cbow=True,
            epochs=5,
            seed=0,
            workers=1
        ):
    """Create, builds and trains a gensim word2vec model."""
    model = Word2Vec(
            sentences=sentences,
            vector_size=vector_size,
            min_count=min_count,
            window=window,
            negative=negative,
            sg=0 if cbow else 1,
            epochs=epochs,
            seed=seed,
            workers=workers
        )
    model.build_vocab(sentences)

    model.train(
            sentences,
            total_examples=model.corpus_count,
            epochs=model.epochs
        )
    return model
