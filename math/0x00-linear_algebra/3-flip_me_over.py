#!/usr/bin/env python3
def matrix_transpose(matrix):
    n = len(matrix)
    m = len(matrix[0])
    return [[matrix[j][i] for j in range(n)] for i in range(m)]
