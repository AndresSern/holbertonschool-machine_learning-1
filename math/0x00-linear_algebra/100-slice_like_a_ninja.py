#!/usr/bin/env python3
""" slice fn"""


def np_slice(matrix, axes={}):
    """ slice fn  the * to convert tuplt to slice """
    z = list(axes.keys())[-1]
    z = z + 1
    p = []
    mat = []
    for a in range(z):
        if a in list(axes.keys()):
            p.append(slice(*axes[a]))
        else:
            p.append(slice(None, None, None))
    mat = matrix[p]
    return (mat)
