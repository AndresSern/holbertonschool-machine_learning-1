#!/usr/bin/env python3
"""calculates the expectation step in the EM algorithm for a GMM"""
import numpy as np
pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """
    -X is a numpy.ndarray of shape (n, d) containing
        the data set
    -pi is a numpy.ndarray of shape (k,) containing
        the priors for each cluster
    -m is a numpy.ndarray of shape (k, d) containing
        the centroid means for each cluster
    -S is a numpy.ndarray of shape (k, d, d) containing
        the covariance matrices for each cluster
    Returns:
    g, l, or None, None on failure
    -g is a numpy.ndarray of shape (k, n) containing
        the posterior probabilities for each data point in each cluster
    -l is the total log likelihood

    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None
    if type(pi) is not np.ndarray or len(pi.shape) != 1:
        return None, None
    if not np.isclose(pi.sum(), 1):
        return None, None
    if type(m) is not np.ndarray or len(m.shape) != 2:
        return None, None
    if type(S) is not np.ndarray or len(S.shape) != 3:
        return None, None
    k = pi.shape[0]
    if (k, X.shape[1]) != m.shape or (k, X.shape[1], X.shape[1]) != S.shape:
        return None, None
    K, d = m.shape
    n, d = X.shape
    pdf_prior = np.zeros((K, n))
    for i in range(K):
        PDF = pdf(X, m[i], S[i])
        pdf_prior[i] = PDF * pi[i]
    evidence = np.sum(pdf_prior, axis=0)
    posterior = pdf_prior / evidence
    total_log_likelihood = np.sum(np.log(evidence))
    return posterior, total_log_likelihood
