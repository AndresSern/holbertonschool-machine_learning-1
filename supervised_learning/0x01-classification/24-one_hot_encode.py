#!/usr/bin/env python3
""" one hot encode """

import numpy as np


def one_hot_encode(Y, classes):
    """ ONE HOT ENCODE"""
    if not isinstance(Y, np.ndarray)or classes < 3:
        return None
    if Y.size == 0 or classes is None:
        return None
    Y = Y.astype(int)
    res = np.eye(classes)[Y]
    a, b = res.shape
    if a != classes and b != Y.size:
        return None
    return res
