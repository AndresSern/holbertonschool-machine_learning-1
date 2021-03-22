#!/usr/bin/env python3
import numpy as np
"""
calculates the mean and covariance of a data set
"""


def mean_cov(X):
    """
    Args:
    X is a numpy.ndarray of shape (n, d) containing the data set
    If X is not a 2D numpy.ndarray, raise a TypeError with the message
        X must be a 2D numpy.ndarray
    If n is less than 2, raise a ValueError with the message X must
         contain multiple data points
    Return:
        he mean and covariance of a data set
    """

    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")
    if X.shape[0] < 2:
        raise ValueError("X must contain multiple data points")
    n, d = X.shape
    mean = np.mean(X, axis=0)
    return np.mean(X, axis=0), np.matmul((X - mean).T, X - mean) / (n - 1)
