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
            self.__W = np.random.normal(size=(1, nx))
            self.__b = 0
            self.__A = 0

    @property
    def W(self):
        """The weights vector for the neuron"""
        return self.__W

    @property
    def b(self):
        """The bias for the neuron """
        return self.__b

    @property
    def A(self):
        """ The activated output of the neuron (prediction) """
        return self.__A

    def forward_prop(self, X):
        """ Neuron Forward Propagation
        X is a numpy.ndarray with shape (nx, m) that contains the input data
        """
        z = np.matmul(self.W, X) + self.__b
        Sigmoid_a = 1 / (1 + np.exp(-z))
        self.__A = Sigmoid_a
        return self.__A
