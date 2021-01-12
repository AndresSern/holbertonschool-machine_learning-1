#!/usr/bin/env python3
"""
Returns: the mean and standard
deviation of each feature, respectively
"""
import numpy as np


def normalization_constants(X):
    """
    the mean and standard
    deviation
    """
    return np.mean(X, axis=0), np.std(X, axis=0)
