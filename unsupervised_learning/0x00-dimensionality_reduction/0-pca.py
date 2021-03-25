#!/usr/bin/env python3
"""  performs PCA on a dataset"""

import numpy as np


def pca(X, var=0.95):
    """
    ARGS:
    -X is a numpy.ndarray of shape (n, d) where:
        -n is the number of data points
        -d is the number of dimensions in each point

    all dimensions have a mean of 0 across all data points

    -var is the fraction of the variance that the PCA
    transformation should maintain

    Returns:
    -the weights matrix, W, that maintains var fraction of X‘s
    original variance
    """
    u, s, v = np.linalg.svd(X)
    mean = np.cumsum(s) / np.sum(s)

    ndim = np.where(mean < var, 1, 0)

    ndim = np.sum(ndim)

    return v[:ndim+1, :].T
