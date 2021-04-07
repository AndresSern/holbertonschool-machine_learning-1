#!/usr/bin/env python3
"""
determines the steady state probabilities
of a regular markov chain
"""
import numpy as np


def regular(P):
    """
    ARGS:
        *P is a is a square 2D numpy.ndarray of shape (n, n)
            representing the transition matrix
        *P[i, j] is the probability of transitioning
            from state i to state j
        *n is the number of states in
            the markov chain
    Returns:
        *a numpy.ndarray of shape (1, n) containing the steady
            state probabilities, or None on failure
    """
    P1 = np.matmul(P, P)
    for i in range(2):
        comparison = P == P1
        equal_arrays = comparison.all()
        if equal_arrays:
            return None

        P1 = np.matmul(P1, P)
    if np.any(P1 == 0):
        return None

    e_vals, e_vecs = np.linalg.eig(P.T)
    e_vec1 = e_vecs[:, np.isclose(e_vals, 1)]
    e_vec1 = e_vec1[:, 0]
    stationary = e_vec1 / e_vec1.sum()
    stationary = stationary.real
    return stationary[np.newaxis, :]
