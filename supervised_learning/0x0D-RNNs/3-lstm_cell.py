#!/usr/bin/env python3
"""LSTM unit"""
import numpy as np


class LSTMCell:
    """LSTM unit"""
    def __init__(self, i, h, o):
        """
        -i is the dimensionality of the data
        -h is the dimensionality of the hidden state
        -o is the dimensionality of the outputs
        -Wf, Wu, Wc, Wo, Wy, bf, bu, bc, bo, by :weights and biases of the cell
            -Wf and bf are for the forget gate
            -Wu and bu are for the update gate
            -Wc and bc are for the intermediate cell state
            -Wo and bo are for the output gate
            -Wy and by are for the outputs
        """
        self.Wf = np.random.normal(size=(i+h, h))
        self.Wu = np.random.normal(size=(i+h, h))
        self.Wc = np.random.normal(size=(i+h, h))
        self.Wo = np.random.normal(size=(i+h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def softmax(self, z):
        """ softmax function"""
        t = np.exp(z - np.max(z))
        sf = t / t.sum(axis=1, keepdims=True)
        return sf

    def sigmoid(self, x_t):
        """ sigmoid function"""
        return 1/(1+np.exp(-x_t))

    def forward(self, h_prev, c_prev, x_t):
        """
        ARGS:
        x_t :{numpy.ndarray} shape (m, i) : input for the cell
            m is the batche size for the data
        h_prev: {numpy.ndarray}  shape (m, h) :the previous hidden state
        c_prev: {numpy.ndarray} shape (m, h) :the previous cell state

        Returns: h_next, c_next, y
            h_next is the next hidden state
            c_next is the next cell state
            y is the output of the cell
        """

        # GRU Layer
        sigmoid_input = np.concatenate((h_prev, x_t), axis=1)
        ft = self.sigmoid(np.dot(sigmoid_input, self.Wf) + self.bf)
        ut = self.sigmoid(np.dot(sigmoid_input, self.Wu) + self.bu)

        ct = np.tanh(np.dot(sigmoid_input, self.Wc) + self.bc)
        c_next = c_prev * ft + ut * ct

        ot = self.sigmoid(np.dot(sigmoid_input, self.Wo) + self.bo)

        h_next = ot * np.tanh(c_next)
        # Final output calculation
        y = self.softmax(np.dot(h_next, self.Wy) + self.by)

        return h_next, c_next, y
