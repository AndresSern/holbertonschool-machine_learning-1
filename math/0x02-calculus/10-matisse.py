#!/usr/bin/env python3
""" Polynomial derivative"""


def poly_derivative(poly):
    """
    exp f(x) = x^3 + 3x +5 is [5, 3, 0, 1]
    and f`(x) is [3,0,3]
    """
    p = []
    if poly is None or not isinstance(poly, list) or not poly:
        return None
    if not all(isinstance(n, int) for n in poly):
        return None
    if len(poly) == 1:
        return [0]
    for i in poly:
        if poly.index(i) == 1:
            p.append(poly[1])
        if poly.index(i) > 0 and poly.index(i) != 1:
            p.append(i*poly.index(i))
    test = 1
    for i in p:
        if i != 0:
            test = 0
    if test == 1:
        return [0]
    return(p)
