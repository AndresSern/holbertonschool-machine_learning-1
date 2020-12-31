#!/usr/bin/env python3
""" one hot encode """

import numpy as np


def one_hot_encode(Y, classes):
    """ ONE HOT ENCODE"""
    if not isinstance(Y, np.ndarray)or classes < 3:
        return None
    if Y.size == 0:
        return None
    encode = np.zeros((Y.size, classes))
    encode[np.arange(Y.size), Y] = 1
    res = encode.T
    return res
