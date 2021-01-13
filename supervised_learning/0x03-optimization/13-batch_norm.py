#!/usr/bin/env python3
"""
Z is a numpy.ndarray of shape (m, n) that should be normalized
m is the number of data points
n is the number of features in Z
gamma is a numpy.ndarray of shape (1, n) containing
the scales used for batch normalization
beta is a numpy.ndarray of shape (1, n) containing
the offsets used for batch normalization
epsilon is a small number used to avoid division by zero
Returns: the normalized Z matrix
"""
import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """  batch normalization"""
    m = Z.shape[0]
    u = 1/m * np.sum(Z, axis=0)
    g = 1/m * np.sum((Z - u)**2, axis=0)
    z = (Z - u) / np.sqrt(g + epsilon)
    return z * gamma + beta
