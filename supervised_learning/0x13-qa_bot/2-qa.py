#!/usr/bin/env python3
"""
a function def answer_loop(reference):
that answers questions from a reference text:
"""


def answer_loop(reference):
    """
    ARG:
        -reference is the reference text
    """
    while True:
        txt = input("Q: ")
        if txt in ["exit", "quit", "goodbye", "bye"]:
            print("A: Goodbye")
            break
        a = question_answer(txt, reference)
        if a is None:
            return "Sorry, I do not understand your question."
        print("A: ",a)
