#!/usr/bin/env python3
"""Module for calculating ngram BLEU score."""

from collections import Counter
import numpy as np


def ngram_bleu(references, sentence, n):
    """Create module for calculating ngram BLEU score."""
    c = len(sentence)

    # Récupère la référence la plus proche de la sentence en terme de longueur
    ref_lengths = []
    for ref in references:
        ref_lengths.append(len(ref))

    closest = min(ref_lengths, key=lambda ref_len: abs(ref_len - c))

    # Calcule le BP

    if c > closest:
        BP = 1
    else:
        BP = np.exp(1 - closest / c)

    candidate_ngrams = []
    for i in range(len(sentence) - n + 1):
        ngram = tuple(sentence[i:i+n])
        candidate_ngrams.append(ngram)

    if len(candidate_ngrams) == 0:
        return 0.0

    # Récupère le nombre d'occurences des mots de chaque références

    candidate_counts = Counter(candidate_ngrams)
    max_ref_counts = Counter()

    for ref in references:
        ref_ngrams = []
        for i in range(len(ref) - n + 1):
            ngram = tuple(ref[i:i+n])
            ref_ngrams.append(ngram)

        ref_counts = Counter(ref_ngrams)
        for ngram in ref_counts:
            max_ref_counts[ngram] = max(
                max_ref_counts[ngram], ref_counts[ngram]
            )

    # Compte chaque mot de la phrase plafonné par son occurrence max
    # dans les références (0 si absent)

    clipped_count = 0
    for word in candidate_counts:
        clipped_count += min(candidate_counts[word], max_ref_counts[word])

    precision = clipped_count / len(candidate_ngrams)

    bleu_score = BP * precision

    return bleu_score
