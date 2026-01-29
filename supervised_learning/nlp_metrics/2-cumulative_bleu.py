#!/usr/bin/env python3
"""Module for calculating ngram BLEU score."""

from collections import Counter
import numpy as np


def cumulative_bleu(references, sentence, n):
    """Create module for calculating cumulative ngram BLEU score."""
    c = len(sentence)

    ref_lengths = []
    for ref in references:
        ref_lengths.append(len(ref))
    closest = min(ref_lengths, key=lambda ref_len: abs(ref_len - c))

    if c > closest:
        BP = 1
    else:
        BP = np.exp(1 - closest / c)

    precisions = []

    for i in range(1, n + 1):
        candidate_ngrams = []

        for j in range(len(sentence) - i + 1):
            ngram = tuple(sentence[j:j + i])
            candidate_ngrams.append(ngram)

        if len(candidate_ngrams) == 0:
            return 0.0

        candidate_counts = Counter(candidate_ngrams)
        max_ref_counts = Counter()

        for ref in references:
            ref_ngrams = []

            for k in range(len(ref) - i + 1):
                ngram = tuple(ref[k:k + i])
                ref_ngrams.append(ngram)

            ref_counts = Counter(ref_ngrams)

            for ngram in ref_counts:
                max_ref_counts[ngram] = max(
                    max_ref_counts[ngram], ref_counts[ngram]
                )

        clipped_count = 0
        for word in candidate_counts:
            clipped_count += min(candidate_counts[word], max_ref_counts[word])

        precision = clipped_count / len(candidate_ngrams)
        precisions.append(precision)

    if min(precisions) == 0:
        return 0.0

    product = 1
    for p in precisions:
        product *= p

    geometric_mean = product ** (1 / n)
    bleu_score = BP * geometric_mean

    return bleu_score
