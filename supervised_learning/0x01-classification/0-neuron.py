#!/usr/bin/env python3
"""
the Neuron
"""

import numpy as np


class Neuron:
    """ the Neuron """
    def __init__(self, nx):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        elif nx <= 0:
            raise ValueError("nx must be a positive integer")
        else:
            self.W = np.random.normal(size=(1, nx))
            self.b = 0
            self.A = 0
