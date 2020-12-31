#!/usr/bin/env python3
""" one hot encode """

import numpy as np


def one_hot_encode(Y, classes):
    if not isinstance(Y, np.ndarray)or classes < 1:
        return None
    if Y.all() is None:
        return None
    encode = np.zeros((Y.size, classes))
    encode[np.arange(Y.size), Y] = 1
    return encode.T
