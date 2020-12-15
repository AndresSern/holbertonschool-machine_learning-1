#!/usr/bin/env python3
"""
Initialize Exponential
"""


class Exponential:
    """ calculate lambtha """
    def __init__(self, data=None, lambtha=1.):
        self.data = data
        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
            else:
                self.lambtha = float(lambtha)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 3:
                raise ValueError("data must contain multiple values")
            else:
                S = sum(data)
                lam = len(data) / S
                self.lambtha = float(lam)

    def pdf(self, x):
        """ probability density function"""
        if x < 0:
            return 0
        e = 2.7182818285
        pdf = (self.lambtha) * (e ** ((self.lambtha) * - 1 * x))
        return pdf

    def cdf(self, x):
        """  cumulative distribution function"""
        if x < 0:
            return 0
        e = 2.7182818285
        cdf = 1 - (e ** ((self.lambtha) * - 1 * x))
        return cdf
