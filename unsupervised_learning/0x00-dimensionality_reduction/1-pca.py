#!/usr/bin/env python3
"""  performs PCA on a dataset"""

import numpy as np


def pca(X, ndim):
    """
    ARGS:
    -X is a numpy.ndarray of shape (n, d) where:
        -n is the number of data points
        -d is the number of dimensions in each point

    all dimensions have a mean of 0 across all data points

    -ndim is the new dimensionality of the transformed X

    Returns:
    -Returns: T, a numpy.ndarray of shape (n, ndim) containing
    the transformed version of X
    """
    '''center columns by subtracting column means.'''
    C = X - np.mean(X, axis=0)
    u, s, v = np.linalg.svd(C)
    w = v[:ndim].T
    return np.matmul(C, w)
