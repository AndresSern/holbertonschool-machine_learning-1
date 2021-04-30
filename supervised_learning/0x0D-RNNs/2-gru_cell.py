#!/usr/bin/env python3
""" represents a gated recurrent unit:"""
import numpy as np


class GRUCell:
    """ represents a gated recurrent unit:"""
    def __init__(self, i, h, o):
        """
        -i is the dimensionality of the data
        -h is the dimensionality of the hidden state
        -o is the dimensionality of the outputs
        - Wz, Wr, Wh, Wy, bz, br, bh, by : weights and biases of the cell
            -Wz and bz are for the update gate
            -Wr and br are for the reset gate
            -Wh and bh are for the intermediate hidden state
            -Wy and by are for the output
        """
        self.Wz = np.random.normal(size=(i+h, h))
        self.Wr = np.random.normal(size=(i+h, h))
        self.Wh = np.random.normal(size=(i+h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def softmax(self, z):
        """ softmax function"""
        t = np.exp(z - np.max(z))
        sf = t / t.sum(axis=1, keepdims=True)
        return sf

    def sigmoid(self, x_t):
        """ sigmoid function"""
        return 1/(1+np.exp(-x_t))

    def forward(self, h_prev, x_t):
        """
        ARGS:
            -x_t: {numpy.ndarray} shape (m, i) : data input for the cell
                -m is the batche size for the data
            -h_prev: {numpy.ndarray} shape (m, h)
                 containing the previous hidden state

        Returns: h_next, y
            -h_next is the next hidden state
            -y is the output of the cell
        """
        # GRU Layer
        sigmoid_input = np.concatenate((h_prev, x_t), axis=1)
        zt = self.sigmoid(np.dot(sigmoid_input, self.Wz) + self.bz)
        rt = self.sigmoid(np.dot(sigmoid_input, self.Wr) + self.br)

        tanh_input = np.concatenate((rt * h_prev, x_t), axis=1)
        ht = np.tanh(np.dot(tanh_input, self.Wh) + self.bh)

        h_next = h_prev * (1 - zt) + zt * ht

        # Final output calculation
        y = self.softmax(np.dot(h_next, self.Wy) + self.by)

        return h_next, y
