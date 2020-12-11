#!/usr/bin/env python3
"""
that calculates the integral of a polynomial:

"""


def poly_integral(poly, C=0):
    """
    exp f(x) = x^3 + 3x +5 is [5, 3, 0, 1]
    give [0, 5, 1.5, 0, 0.25]
    """
    p = [C]
    if not isinstance(poly, list) or not poly:
        return None
    if not all(isinstance(n, int) for n in poly):
        return None

    if not isinstance(C, int):
        return None
    if len(poly) == 1:
        return [C]
    for i in range(len(poly)):
        m = poly[i] / (i + 1)
        if int(m) == m:
            p.append(int(m))
        else:
            p.append(m)
    return p
