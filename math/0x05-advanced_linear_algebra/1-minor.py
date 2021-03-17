#!/usr/bin/env python3
"""
calculates the minor matrix of a matrix
"""


def getMatrixMinor(m, i, j):
    """ get matrix minor"""
    return [row[:j] + row[j + 1:] for row in (m[:i]+m[i+1:])]


def determinant(matrix):
    """ calculates the determinant of a matrix"""

    if len(matrix) == 0 or not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")
    if (len(matrix[0]) == 0):
        return 1
    if len(matrix[0]) != len(matrix):
        raise ValueError("matrix must be a square matrix")
    for item in matrix:
        if not isinstance(item, list):
            raise TypeError("matrix must be a list of lists")

    if (len(matrix[0]) == 1):
        return matrix[0][0]

    """base case for 2x2 matrix"""
    if len(matrix) == 2:
        return matrix[0][0]*matrix[1][1]-matrix[0][1]*matrix[1][0]

    det = 0
    for c in range(len(matrix)):
        det += ((-1)**c)*matrix[0][c]*determinant(getMatrixMinor(matrix, 0, c))
    return det


def minor(matrix):
    """
    calculates the minor matrix of a matrix
    """

    if not isinstance(matrix, list):
        raise TypeError("matrix must be a list of lists")
    if not matrix:
        raise TypeError("matrix must be a list of lists")
    for item in matrix:
        if len(item) != len(matrix):
            raise ValueError("matrix must be a non-empty square matrix")
        if not isinstance(item, list):
            raise TypeError("matrix must be a list of lists")
    if (len(matrix[0]) == 1):
        return [[1]]
    h = []
    for i in range(len(matrix)):

        for c in range(len(matrix)):
            a = determinant(getMatrixMinor(matrix, i, c))
            h.append(a)
    rst = []
    for i in range(0, len(h), len(matrix)):
        rst.append(h[i: i + len(matrix)])
    return rst
