#!/usr/bin/env python3
""" one hot encode """

import numpy as np


def one_hot_encode(Y, classes):
    """ ONE HOT ENCODE"""
    if isinstance(Y, np.ndarray)or classes > 3:
        if Y.size != 0:
            encode = np.zeros((Y.size, classes))
            encode[np.arange(Y.size), Y] = 1
            res = encode.T
            return res
        else:
            return None
    else:
        return None
