#!/usr/bin/env python3
"""bidirectional RNN"""
import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """
    forward propagation for a bidirectional RNN

    ARGS:
        -bi_cell : {instance of BidirectinalCell}
        -X: {numpy.ndarray} of shape (t, m, i): data to be used
            -t is the maximum number of time steps
            -m is the batch size
            -i is the dimensionality of the data
        -h_0: {numpy.ndarray shape (m, h)}
            the initial hidden state in the forward direction
        -h is the dimensionality of the hidden state
        -h_t : {numpy.ndarray of shape (m, h)} is
            the initial hidden state in the backward direction
    Returns: H, Y
        H :{numpy.ndarray} containing all of the concatenated hidden states
        Y :{numpy.ndarray} containing all of the outputs
    """
    t, _, i = X.shape
    Hf = []
    Hb = []
    Hsb = h_t
    h_prev = h_0

    for i in range(t):
        h_prev = bi_cell.forward(h_prev, X[i])
        Hsb = bi_cell.backward(Hsb, X[t-i-1])
        Hf.append(h_prev)
        Hb.append(Hsb)

    Hb = [i for i in reversed(Hb)]

    H = np.concatenate((np.array(Hf), np.array(Hb)), axis=-1)
    Y = bi_cell.output(H)
    return H, Y
