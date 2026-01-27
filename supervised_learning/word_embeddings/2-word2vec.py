#!/usr/bin/env python3
"""Word to vector."""

import gensim


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
    model = gensim.models.Word2Vec(
            sentences=sentences,
            vector_size=vector_size,
            window=window,
            min_count=min_count,
            workers=workers,
            seed=seed,
            epochs=epochs,
            sg=0 if cbow else 1,
            negative=negative
        )

    return model
