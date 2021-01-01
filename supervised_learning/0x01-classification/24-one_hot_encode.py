#!/usr/bin/env python3
""" one hot encode """

import numpy as np


def one_hot_encode(Y, classes):
    """ ONE HOT ENCODE"""
    if not isinstance(Y, np.ndarray)or classes < 3:
        return None
    if Y.size == 0 or classes is None:
        return None
    if Y.ndim != 1:
        return None
    res = np.eye(classes)[Y]
    res = res.T
    a, b = res.shape
    if a != classes and b != Y.size:
        return None
    return res
