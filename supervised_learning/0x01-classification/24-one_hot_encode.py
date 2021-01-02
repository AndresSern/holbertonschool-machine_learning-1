#!/usr/bin/env python3
""" one hot encode """

import numpy as np


def one_hot_encode(Y, classes):
    """ ONE HOT ENCODE
    Y is a numpy.ndarray containing numeric class labels
    m is the number of examples
    classes is the maximum number of classes found in Y
    np.eye
    """
    if not isinstance(Y, np.ndarray):
        return None
    try:
        encode = np.zeros((Y.size, classes))
        encode[np.arange(Y.size), Y] = 1
        res = encode.T
        return res
    except Exception:
        return None
