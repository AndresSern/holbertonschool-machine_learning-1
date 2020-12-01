#!/usr/bin/env python3
def add_matrices2D(mat1, mat2):
    arrr = []
    if len(mat1) != len(mat2) or len(mat1[0]) != len(mat2[0]):
        return None
    for i in range(len(mat1)):
        arr = []
        for j in range(len(mat1[0])):
            arr.append(mat1[i][j] + mat2[i][j])
        arrr.append(arr)
    return arrr
