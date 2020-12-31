#!/usr/bin/env python3
""" one hot decode """

import numpy as np


def one_hot_decode(one_hot):
    if not isinstance(one_hot, np.ndarray):
        return None
    lst = []
    for i in range(len(one_hot)):
        for j in range(len(one_hot[i])):
            if one_hot[i][j] == 1:
                lst.append(j)
    arr = np.array(lst)
    return arr
