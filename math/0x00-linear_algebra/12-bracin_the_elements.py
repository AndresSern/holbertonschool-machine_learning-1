#!/usr/bin/env python3
""" np opp"""


def np_elementwise(mat1, mat2):
    """np opp"""
    add = mat1 + mat2
    diff = mat1 - mat2
    mul = mat1 * mat2
    div = mat1 / mat2
    mytuple = (add, diff, mul, div)
    return mytuple
