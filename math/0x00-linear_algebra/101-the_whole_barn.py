#!/usr/bin/env python3
""" add fn"""


import numpy as np


def add_matrices(mat1, mat2):
    """ add fn"""
    m1 = np.array(mat1)
    m2 = np.array(mat2)
    if m1.shape != m2.shape:
        return None
    a = np.add(m1, m2)
    m = []
    for x in a:
        list1 = x.tolist()
        m.append(list1)
    return (m)
