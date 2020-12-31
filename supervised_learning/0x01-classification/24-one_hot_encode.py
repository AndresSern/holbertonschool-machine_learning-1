#!/usr/bin/env python3
""" one hot encode """

import numpy as np


def one_hot_encode(Y, classes):
    """ ONE HOT ENCODE"""
    if not isinstance(Y, np.ndarray)or classes < 3:
        return None
    if Y[0] is None or len(Y) < 1:
        return None
    encode = np.zeros((Y.size, classes))
    encode[np.arange(Y.size), Y] = 1
    res = encode.T
    a = Y.shape[0]
    aa, bb = res.shape
    if aa != classes or bb != a:
        return None
    return res
