#!/usr/bin/env python3
"""
Bag Of Words

creates a bag of words embedding matrix
"""
from sklearn.feature_extraction.text import CountVectorizer


def bag_of_words(sentences, vocab=None):
    """
    ARGS:
        *sentences :{list} :sentences to analyze
        *vocab: {list} : vocabulary words for the analysis

    Returns: embeddings, features
        *embeddings: {numpy.ndarray} : shape (s, f) => embeddings
            -s is the number of sentences in sentences
            -f is the number of features analyzed
        *features: {list} : of the features used for embeddings
    """
    vectorizer2 = CountVectorizer(analyzer='word', vocabulary=vocab)
    X2 = vectorizer2.fit_transform(sentences)
    embeddings = X2.toarray()
    features = vectorizer2.get_feature_names()
    return embeddings, features
