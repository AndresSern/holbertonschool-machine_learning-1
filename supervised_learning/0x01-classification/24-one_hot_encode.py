#!/usr/bin/env python3
""" one hot encode """

import numpy as np


def one_hot_encode(Y, classes):
    """ ONE HOT ENCODE
    Y is a numpy.ndarray containing numeric class labels
    m is the number of examples
    classes is the maximum number of classes found in Y
    """
    if not isinstance(Y, np.ndarray)or classes < 3:
        return None
    if Y.size == 0 or classes is None:
        return None
    try:
        res = np.eye(classes)[Y]
        res = res.T
        a, b = res.shape
        if a != classes and b != Y.size:
            return None
        return res
    except Exception:
        return None
