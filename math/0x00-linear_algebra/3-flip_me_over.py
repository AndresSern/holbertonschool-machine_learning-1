#!/usr/bin/env python3
""" transpose"""


def matrix_transpose(matrix):
    """ transpose"""
    n = len(matrix)
    m = len(matrix[0])
    return [[matrix[j][i] for j in range(n)] for i in range(m)]
