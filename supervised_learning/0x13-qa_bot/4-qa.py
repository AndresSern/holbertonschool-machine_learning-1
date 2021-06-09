#!/usr/bin/env python3
"""
 Multi-reference Question Answering
"""

question_answer1 = __import__('0-qa').question_answer
semantic_search = __import__('3-semantic_search').semantic_search


def question_answer(corpus_path):
    """
    Answers questions from multiple reference texts:
    ARGS:
        corpus_path is the path to the corpus of reference documents
    """
    while True:
        txt = input("Q: ")
        if txt in ["exit", "quit", "goodbye", "bye"]:
            print("A: Goodbye")
            break

        reference = semantic_search(corpus_path, txt)
        answer = question_answer1(txt, reference)
        if answer is None:
            return "Sorry, I do not understand your question."
        print("A:", answer)
