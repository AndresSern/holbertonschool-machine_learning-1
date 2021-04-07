#!/usr/bin/env python3
"""
determines if a markov chain is absorbing
"""
import numpy as np


def update_index(index, j, P):
    """update_index"""
    X = P[:, j]
    for i in range(P.shape[0]):
        if X[i] > 0:
            index.append(i)
    return index


def absorbing(P):
    """
    ARGS:
        *P is a is a square 2D numpy.ndarray of shape
            (n, n) representing the standard transition matrix
            -P[i,j] is the probability of transitioning from state i to state j
            -n is the number of states in the markov chain
    Returns:
        *True if it is absorbing, or False on failure
    """
    diag = np.diagonal(P)

    if not diag.any() == 1:
        return False

    if np.all(diag == 1):
        return True
    index = np.where(diag == 1)[0]
    index = list(index)
    if not index:
        return False
    for i in range(P.shape[0]):
        if i in index:
            index = update_index(index, i, P)

    for i in range(P.shape[0]):
        if i in index:
            index = update_index(index, i, P)
    return len(set(index)) == P.shape[0]
