#!/usr/bin/env python3
"""summation fn"""


def summation_i_squared(n):
    """summation fn"""
    if n is None or n <= 0:
        return(None)
    return (int)(n * (n + 1) * (2 * n + 1) / 6)
