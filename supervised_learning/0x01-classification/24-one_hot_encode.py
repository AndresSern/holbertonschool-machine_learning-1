#!/usr/bin/env python3
""" one hot encode """

import numpy as np


def one_hot_encode(Y, classes):
    """ ONE HOT ENCODE"""
    if not isinstance(Y, np.ndarray)or classes < 3:
        return None
    if Y.size == 0 or classes is None:
        return None
    encode = np.zeros((Y.size, classes))
    encode[np.arange(Y.size), Y] = 1
    res = encode.T
    a, b = res.shape
    if a != classes and b != Y.size:
        return None
    return res
