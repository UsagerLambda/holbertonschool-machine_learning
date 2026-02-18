#!/usr/bin/env python3
"""Boucle de question réponses sans réponses."""


def start_loop():
    """Boucle de question réponses sans réponses."""
    while True:
        user_input = input("Q: ")

        if user_input.lower() in ["exit", "quit", "goodbye", "bye"]:
            print("A: Goodbye")
            break

        print("A: ")


if __name__ == '__main__':
    start_loop()
