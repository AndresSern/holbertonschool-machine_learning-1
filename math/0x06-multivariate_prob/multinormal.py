#!/usr/bin/env python3
"""
Multivariate Normal distribution
"""
import numpy as np


class MultiNormal:
    """
    multiNormal
    """
    def __init__(self, data):
        """
        Multivariate Normal distribution
        Set the public instance
        """
        if not isinstance(data, np.ndarray):
            raise TypeError("data must be a 2D numpy.ndarray")
        if len(data.shape) != 2:
            raise TypeError("data must be a 2D numpy.ndarray")
        if data.shape[1] < 2:
            raise ValueError("data must contain multiple data points")
        d, n = data.shape
        self.mean = np.mean(data, axis=1, keepdims=True)
        c = np.matmul((data - self.mean), (data - self.mean).T) / (n - 1)
        self.cov = c

    def pdf(self, x):
        """ calculates the PDF at a data point"""

        if not isinstance(x, np.ndarray):
            raise TypeError('x must be a numpy.ndarray')

        d = self.cov.shape[0]
        if len(x.shape) != 2:
            raise ValueError('x must have the shape ({}, 1)'.format(d))
        if x.shape[1] != 1 or x.shape[0] != d:
            raise ValueError('x must have the shape ({}, 1)'.format(d))
        n, e = x.shape
        d = self.mean.shape[0]
        x_m = x - self.mean
        v1 = (np.sqrt((2 * np.pi) ** d * np.linalg.det(self.cov)))
        v2 = np.exp(-(np.linalg.solve(self.cov, x_m).T.dot(x_m)) / 2)
        return (1 / v1 * v2)
