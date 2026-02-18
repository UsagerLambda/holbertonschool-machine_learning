#!/usr/bin/env python3
"""Boucle de question réponses."""

question_answer = __import__('0-qa').question_answer


def answer_loop(reference):
    """Boucle de question réponses."""
    while True:
        user_input = input("Q: ")
        if not user_input.strip():
            continue

        if user_input.lower() in ["exit", "quit", "goodbye", "bye"]:
            print("A: Goodbye")
            break

        answer = question_answer(user_input, reference)
        if answer:
            print(f"A: {answer}")
        else:
            print("A: Sorry, I do not understand your question.")
