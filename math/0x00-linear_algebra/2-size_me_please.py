#!/usr/bin/env python3
""" shape"""


def matrix_shape(matrix):
    mat1 = []
    while isinstance(matrix, list):
        mat1.append(len(matrix))
        matrix = matrix[0]
    return mat1
