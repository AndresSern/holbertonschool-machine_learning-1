#!/usr/bin/env python3
"""
calculates the mean and covariance of a data set
"""
import numpy as np


def mean_cov(X):
    """ mean and covariance"""
    if not isinstance(X, np.ndarray):
        raise TypeError("X must be a 2D numpy.ndarray")
    if len(X.shape) != 2:
        raise TypeError("X must be a 2D numpy.ndarray")
    if X.shape[0] < 2:
        raise ValueError("X must contain multiple data points")
    n, d = X.shape
    mean = np.mean(X, axis=0, keepdims=True)
    conv = np.matmul((X - mean).T, (X - mean)) / (n - 1)
    return mean, conv
