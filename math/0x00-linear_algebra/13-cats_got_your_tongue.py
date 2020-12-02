#!/usr/bin/env python3
""" np concate"""
import numpy as np


def np_cat(mat1, mat2, axis=0):
    """ np concate"""
    ar = np.concatenate((mat1, mat2), axis=axis)
    return ar
