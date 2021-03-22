#!/usr/bin/env python3
import numpy as np
"""
calculates the mean and covariance of a data set
"""


def mean_cov(X):
    """
    Args:
    X is a numpy.ndarray of shape (n, d) containing the data set
    Return:
        he mean and covariance of a data set
    """

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")
    if X.shape[0] < 2:
        raise ValueError("X must contain multiple data points")
    n, d = X.shape
    mean = np.mean(X, axis=0)
    a = X - mean
    return np.mean(X, axis=0), np.matmul(a.T, a) / (n - 1)
