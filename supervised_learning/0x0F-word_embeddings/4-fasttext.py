#!/usr/bin/env python3
"""
FastText

creates and trains a genism fastText model:
"""
from gensim.models import FastText


def fasttext_model(sentences, size=100, min_count=5, negative=5,
                   window=5, cbow=True, iterations=5, seed=0, workers=1):
    """
    ARGS:
        *sentences: {list} : sentences to be trained on
        *size :dimensionality of embedding layer
        *min_count: minimum number of occurrences of a word
        *window:  maximum distance between the current and
                  predicted word within a sentence
        *negative : size of negative sampling
        *cbow: {boolean} :training type; True  CBOW; False  Skip-gram
        *iterations: number of iterations to train over
        *seed: seed for the random number generator
        workers: number of worker threads to train the model
    Returns: the trained model
    """
    model = FastText(sentences, size=size, min_count=min_count, window=window,
                     negative=negative, sg=cbow,
                     seed=seed, workers=workers)

    model.train(sentences, total_examples=model.corpus_count,
                epochs=iterations)

    return model
