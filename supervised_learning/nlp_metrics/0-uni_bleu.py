#!/usr/bin/env python3
"""Module for calculating unigram BLEU score."""

from collections import Counter

import numpy as np


def uni_bleu(references, sentence):
    """
    Calculate the unigram BLEU score for a sentence.

    Args:
        references: list of reference translations (each is a list of words)
        sentence: list of words in the candidate sentence

    Returns:
        The unigram BLEU score
    """
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

    # Récupère le nombre d'occurences des mots de chaque références

    sentence_counts = Counter(sentence)
    max_ref_counts = Counter()

    for ref in references:
        ref_counts = Counter(ref)
        for word in ref_counts:
            max_ref_counts[word] = max(max_ref_counts[word], ref_counts[word])

    # Compte chaque mot de la phrase plafonné par son occurrence max
    # dans les références (0 si absent)

    clipped_counts = 0
    for word in sentence_counts:
        clipped_counts += min(sentence_counts[word], max_ref_counts[word])

    if c == 0:
        precision = 0
    else:
        precision = clipped_counts / c

    bleu_score = BP * precision

    return bleu_score
