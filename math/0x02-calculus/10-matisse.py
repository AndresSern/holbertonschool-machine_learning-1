#!/usr/bin/env python3
""" Polynomial derivative"""


def poly_derivative(poly):
    """
    exp f(x) = x^3 + 3x +5 is [5, 3, 0, 1]
    and f`(x) is [3,0,3]
    """
    p = []
    if not isinstance(poly, list) or not poly:
        return None
    if not all(isinstance(n, int) for n in poly):
        return None
    if len(poly) == 1:
        return [0]
    poly = poly[1:]
    for i in range(len(poly)):
        p.append(poly[i] * (i + 1))
    return p
