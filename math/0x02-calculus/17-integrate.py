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
    if not isinstance(C, int):
        return None
    if len(poly) == 1:
        return [C]
    i = 0
    for item in poly:
        i = i + 1
        m = item / i
        if int(m) == m:
            p.append((int)(m))
        else:
            p.append(m)
    x = 0
    for i in range(len(p)-1, 0):
        if(p[i] == 0 and x == 0):
            p.pop(i)
        else:
            x = 1
    return p
