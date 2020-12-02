#!/usr/bin/env python3
""" cat_matrices"""


def cat_matrices2D(mat1, mat2, axis=0):
    """ cat_matrices"""
    mat = []
    if axis == 1 and len(mat1) == len(mat2):
        for i in range(len(mat1)):
            a = []
            for j in range(len(mat1[0])):
                a.append(mat1[i][j])
            for j in range(len(mat2[0])):
                a.append(mat2[i][j])
            mat.append(a)
        return mat
    if axis == 0 and len(mat1[0]) == len(mat2[0]):
        mat = mat1 + mat2
        return mat
    return None
