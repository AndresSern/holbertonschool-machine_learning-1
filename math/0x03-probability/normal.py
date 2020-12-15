#!/usr/bin/env python3
"""
Initialize Poisson
"""


class Normal:
    """ calculate Normal """
    def __init__(self, data=None, mean=0., stddev=1.):
        self.data = data
        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
            else:
                self.mean = float(mean)
                self.stddev = float(stddev)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")
            if len(data) < 3:
                raise ValueError("data must contain multiple values")
            else:
                S = sum(data)
                mean = S / len(data)
                self.mean = float(mean)
                count = 0
                for i in data:
                    count += (i-mean) ** 2
                std = count / len(data)
                stdd = std ** (1/2)
                self.stddev = float(stdd)

    def z_score(self, x):
        """ z score """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """ x value """
        return (z * self.stddev) + self.mean

    def pdf(self, x):
        """ probability density function """
        π = 3.1415926536
        e = 2.7182818285
        b = ((self.stddev) * ((2 * π) ** (1 / 2)))
        c = ((x - self.mean) / self.stddev)
        pdf = (1 / b) * e ** (((1/2) * -1) * c ** 2)
        return pdf

    def erf(self, x):
        """ erf function"""
        π = 3.1415926536
        z = ((x ** 9) / 216)
        b = x - ((x ** 3) / 3) + ((x ** 5) / 10) - ((x ** 7) / 42) + z
        return (2 / π ** (1/2)) * (b)

    def cdf(self, x):
        """ cdf fn"""
        erf_x_val = (x - self.mean) / (self.stddev * (2 ** (1/2)))
        return (1/2) * (1 + self.erf(erf_x_val))
