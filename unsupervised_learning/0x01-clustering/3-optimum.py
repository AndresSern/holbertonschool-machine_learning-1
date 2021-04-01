#!/usr/bin/env python3
"""
tests for the optimum number of clusters by variance:
"""

import numpy as np
kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """
    ARGS:
    -X is a numpy.ndarray of shape (n, d) containing the data set
    -kmin is a positive integer containing the minimum number
        of clusters to check for (inclusive)
    -kmax is a positive integer containing the maximum number
        of clusters to check for (inclusive)
    -iterations is a positive integer containing the maximum
        number of iterations for K-means
    RETURNS:
    -results, d_vars, or None, None on failure
        -K-means for each cluster size
        -d_vars is a list containing the difference in variance
        from the smallest cluster size for each cluster size
    """
    '''
    if kmax == None:
        kmax = X.shape[0]
    '''
    if (not isinstance(X, np.ndarray) or not isinstance(iterations, int)
            or iterations <= 0
            or len(X.shape) != 2 or not isinstance(kmin, int)
            or kmin < 0
            or not isinstance(kmax, int) or kmax < 0
            or kmax - kmin < 1):
        return None, None
    res = []
    d = []
    c, clas = kmeans(X, kmin, iterations)
    v1 = variance(X, c)
    for i in range(kmin, kmax+1):
        c, clas = kmeans(X, i, iterations)
        a = (c, clas)
        res.append(a)
        v = variance(X, c)
        d.append(v1 - v)
    return res, d
