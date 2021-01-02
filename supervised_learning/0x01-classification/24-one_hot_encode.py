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
    try:
        res = np.eye(classes)[Y]
        res = res.T
        return res
    except Exception:
        return None
