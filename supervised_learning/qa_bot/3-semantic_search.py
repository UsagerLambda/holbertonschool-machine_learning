#!/usr/bin/env python3
"""Trouve le fichier markdown le plus approprié pour répondre."""

import os
import numpy as np
import tensorflow_hub


def semantic_search(corpus_path, sentence):
    """Trouve le fichier markdown le plus approprié pour répondre.

    Args:
        corpus_path (string): Chemin du dossier ou chercher les fichiers
        sentence (string): Question pour la recherche de similarité

    Returns:
        string: contenu du fichier trouvé
    """
    url = "https://www.kaggle.com/models"
    embed = tensorflow_hub.load(
        f"{url}/google/universal-sentence-encoder/TensorFlow2/large/2")

    documents = []

    # récupère et stocke tous les fichiers markdown dans le dossier corpus_path
    for file in os.listdir(corpus_path):
        if file.endswith(".md"):
            with open(f"{corpus_path}/{file}", 'r') as f:
                documents.append(f.read())

    # Embedding des documents et de la question
    embeds_doc = embed(documents)
    embeds_sentence = embed([sentence])

    # Calcul de similaritée
    sim = np.inner(embeds_doc, embeds_sentence)

    # Récupère la dimension (l'indice) la plus similaire
    best_doc = np.argmax(sim)

    # Retourne le contenu du document à l'indice trouvé
    return documents[best_doc]
