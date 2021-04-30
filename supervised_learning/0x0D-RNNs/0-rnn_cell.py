#!/usr/bin/env python3
""" the class RNNCell """
import numpy as np


class RNNCell:
    """  the class RNNCell """
    def __init__(self, i, h, o):
        """
        ARGS:
        -i is the dimensionality of the data

        -h is the dimensionality of the hidden state

        -o is the dimensionality of the outputs

        -Wh,Wy, bh, by :weights and biases of the cell
        -Wh and bh are for the concatenated hidden
            state and input data
        -Wy and by are for the output
        """
        self.Wh = np.random.normal(size=(i+h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def softmax(self, z):
        """ SOFTMAX FUNCTION"""
        t = np.exp(z - np.max(z))
        sf = t / t.sum(axis=1, keepdims=True)
        return sf

    def forward(self, h_prev, x_t):
        """
        ARGS:
            -x_t : {numpy.ndarray)  shape (m, i)
            contains the data input for the cell
                -m is the batche size for the data
            -h_prev {numpy.ndarray} shape (m, h)
            containing the previous hidden state

        Returns: h_next, y
            -h_next is the next hidden state
            -y is the output of the cell
        """
        # RNN Layer
        tanh_input = np.concatenate((h_prev, x_t), axis=1)
        h_next = np.tanh(np.dot(tanh_input, self.Wh) + self.bh)
        # Final output calculation
        yt_pred = self.softmax(np.dot(h_next, self.Wy) + self.by)
        return h_next, yt_pred
