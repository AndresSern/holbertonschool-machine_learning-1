#!/usr/bin/env python3
"""
Initialize Poisson
"""


class Poisson:
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
                a = S / len(data)
                self.lambtha = float(a)

    def pmf(self, k):
        """ probability mass function"""
        if k < 0:
            return 0
        if not isinstance(k, int):
            k = int(k)
        e = 2.7182818285
        fact = 1
        for i in range(1, int(k) + 1):
            fact = fact * i
        a = (e ** ((self.lambtha) * - 1)) * (self.lambtha ** k)
        PMF = a / fact
        return PMF

    def cdf(self, k):
        """  cumulative distribution function"""
        if k < 0:
            return 0
        if not isinstance(k, int):
            k = int(k)
        cdf = 0
        for i in range(0, int(k) + 1):
            cdf += self.pmf(i)
        return cdf
