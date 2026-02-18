#!/usr/bin/env python3
"""Boucle de question réponses++."""

semantic_search = __import__('3-semantic_search').semantic_search
question = __import__('0-qa').question_answer


def question_answer(coprus_path):
    """Boucle de question réponses++.

    Args:
        coprus_path (string): Chemin du dossier de recherche
    """
    while True:
        user_input = input("Q: ")
        if not user_input.strip():
            continue

        if user_input.lower() in ["exit", "quit", "goodbye", "bye"]:
            print("A: Goodbye")
            break

        reference = semantic_search(coprus_path, user_input)
        answer = question(user_input, reference)

        if answer:
            print(f"A: {answer}")
        else:
            print("A: Sorry, I do not understand your question.")
