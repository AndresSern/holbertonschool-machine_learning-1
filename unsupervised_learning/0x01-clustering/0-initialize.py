#!/usr/bin/env python3
""" initializes cluster centroids for K-means"""
import numpy as np


def initialize(X, k):
    """
    ARGS:
        -X is a numpy.ndarray of shape (n, d) containing the dataset
        that will be used for K-means clustering
            n is the number of data points
            d is the number of dimensions for each data point
        -k is a positive integer containing the number of clusters
    RETURNS:
        -numpy.ndarray of shape (k, d) containing the initialized
        centroids for each cluster, or None on failure

    """
    if (not isinstance(X, np.ndarray) or not isinstance(k, int) or k <= 0
            or len(X.shape) != 2):
        return None
    n, d = X.shape
    minn = np.amin(X, axis=0)
    maxx = np.amax(X, axis=0)
    a = np.random.uniform(minn, maxx, (k, d))
    return a
