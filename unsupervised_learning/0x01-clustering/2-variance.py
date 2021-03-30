#!/usr/bin/env python3

""" calculates the total intra-cluster variance for a data set"""
import numpy as np


def variance(X, C):
    """
    ARGS:
        -X is a numpy.ndarray of shape (n, d)
        containing the data set
        -C is a numpy.ndarray of shape (k, d)
        containing the centroid means for each cluster

    Returns: var, or None on failure
        -var is the total variance
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None
    if type(C) is not np.ndarray or len(C.shape) != 2:
        return None
    if (not isinstance(X, np.ndarray) or not isinstance(C, np.ndarray)
            or len(X.shape) != 2 or len(C.shape) != 2
            or X.shape[1] != C.shape[1]):
        return None
    Dist = np.linalg.norm(X - C[:, np.newaxis], axis=2)
    variance = np.sum(np.square(np.min(Dist, axis=0)))
    return variance
