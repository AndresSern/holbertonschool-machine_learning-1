#!/usr/bin/env python3
"""
performs K-means on a dataset:
"""
import numpy as np


def initialize(X, k):
    """
    Initializes cluster centroids for K-means
    """
    if (not isinstance(X, np.ndarray) or not isinstance(k, int) or k <= 0
            or len(X.shape) != 2):
        return None
    n, d = X.shape
    minn = np.amin(X, axis=0)
    maxx = np.amax(X, axis=0)
    a = np.random.uniform(minn, maxx, (k, d))
    return a


def kmeans(X, k, iterations=1000):
    """
    ARGS:
        -X is a numpy.ndarray of shape (n, d) containing the dataset
        -n is the number of data points
        -d is the number of dimensions for each data point
        -k is a positive integer containing the number of clusters
        -iterations is a positive integer containing the maximum number
        of iterations that should be performed

    REturns:
    C, clss, or None, None on failure
        -C is a numpy.ndarray of shape (k, d)
        containing the centroid means for each cluster
        -clss is a numpy.ndarray of shape (n,) containing
        the index of the cluster in C that each data point belongs to
    """

    """ containing the centroid means for each cluster"""
    if (not isinstance(X, np.ndarray) or not isinstance(k, int) or k <= 0
            or len(X.shape) != 2):
        return None, None
    C = initialize(X, k)
    clss = []
    for i in range(iterations):
        prev_C = np.copy(C)
        """ Compute the sum of the squared distance between data
        points and all centroids"""
        Dist = np.linalg.norm(X - C[:, np.newaxis], axis=2)
        """ index of the cluster in C that each data point belongs to"""
        clss = Dist.argmin(axis=0)

        for j in range(k):
            idx = np.argwhere(clss == j)
            if not len(idx):
                C[j] = initialize(X, 1)
            else:
                C[j] = np.mean(X[idx], axis=0)

        if (prev_C == C).all():
            return C, clss
    Dist = np.linalg.norm(X[:, None] - C, axis=2)
    clss = Dist.argmin(axis=0)
    return C, clss
