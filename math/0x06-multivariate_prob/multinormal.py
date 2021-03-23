#!/usr/bin/env python3
"""
Multivariate Normal distribution
"""
import numpy as np


class MultiNormal:
    def __init__(self, data):
        """
        Multivariate Normal distribution
        Set the public instance
        """
        if not isinstance(data, np.ndarray):
            raise TypeError("data must be a 2D numpy.ndarray")
        if len(data.shape) != 2:
            raise TypeError("X must be a 2D numpy.ndarray")
        if data.shape[0] < 2:
            raise ValueError("data must contain multiple data points")
        d, n = data.shape
        self.mean = np.mean(data, axis=1, keepdims=True)
        c = np.matmul((data - self.mean), (data - self.mean).T) / (n - 1)
        self.cov = c
