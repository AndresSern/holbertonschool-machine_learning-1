#!/usr/bin/env python3
def mat_mul(mat1, mat2):
    if len(mat1[0]) != len(mat2):
        return None
    mat = []
    for x in range(len(mat1)):
        a = []
        for i in range(len(mat2[0])):
            sum = 0
            for j in range(len(mat1[0])):
                sum = sum + mat1[x][j] * mat2[j][i]
            a.append(sum)
        mat.append(a)
    return mat
