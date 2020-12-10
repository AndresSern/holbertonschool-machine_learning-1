#!/usr/bin/env python3
""" Polynomial derivative"""


def poly_derivative(poly):
    """
    exp f(x) = x^3 + 3x +5 is [5, 3, 0, 1]
    and f`(x) is [3,0,3]
    """
    p = []
    if poly is None or not isinstance(poly, list):
        return None
    for i in poly:
        if poly.index(i) == 1:
            p.append(poly[1])
        if poly.index(i) > 0 and poly.index(i) != 1:
            p.append(i*poly.index(i))
    if sum(p) == 0:
        return [0]
    return(p)
