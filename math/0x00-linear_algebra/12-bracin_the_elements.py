#!/usr/bin/env python3
import numpy as np


def np_elementwise(mat1, mat2):
    add = np.add(mat1, mat2)
    diff = np.subtract(mat1, mat2)
    mul = np.multiply(mat1, mat2)
    div = np.divide(mat1, mat2)
    mytuple = (add, diff, mul, div)
    return mytuple
