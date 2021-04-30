#!/usr/bin/env python3
""" bidirectional Cell """

import numpy as np


class BidirectionalCell():
    """class bidirectionalcell"""
    def __init__(self, i, h, o):
        """
        -i is the dimensionality of the data
        -h is the dimensionality of the hidden states
        -o is the dimensionality of the outputs
        -Whf, Whb, Wy, bhf, bhb, by t: weights and biases of the cell
            -Whf and bhfare for the hidden states in the forward direction
            -Whb and bhbare for the hidden states in the backward direction
            -Wy and byare for the outputs
        """
        self.Whf = np.random.normal(size=(i + h, h))
        self.Whb = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=((2 * h), o))
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """
        ARGS:
            -x_t :{numpy.ndarray} shape (m, i) :the data input for the cell
                -m is the batch size for the data
            -h_prev {numpy.ndarray} shape (m, h) :the previous hidden state
        Returns: h_next, the next hidden state
        """
        input_tanh = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.dot(input_tanh, self.Whf) + self.bhf)
        return h_next
