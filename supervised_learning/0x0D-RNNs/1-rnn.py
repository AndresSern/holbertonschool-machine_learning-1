#!/usr/bin/env python3
""" performs forward propagation for a simple RNN"""
import numpy as np


def rnn(rnn_cell, X, h_0):
    """
    ARGS:
        -rnn_cell: {instance of RNNCell}
        - X : {numpy.ndarray}  shape (t, m, i) data to be used
            t : the maximum number of time steps
            m : the batch size
            i : the dimensionality of the data
        -h_0:{numpy.ndarray} shape (m, h) :the initial hidden state
            h : the dimensionality of the hidden state
    Returns: H, Y
        H : {numpy.ndarray} containing all of the hidden states
        Y : {numpy.ndarray} containing all of the outputs
    """
    t, m, i = X.shape
    m, h = h_0.shape

    H = np.zeros(shape=(t+1, m, h))
    y_pred = np.zeros(shape=(t, m, rnn_cell.by.shape[1]))

    h_next = h_0

    H[0, :, :] = h_next

    for ti in range(t):
        h_next, yt_pred = rnn_cell.forward(h_next, X[ti, :, :])
        # Save the value of the new "next" hidden state in a (≈1 line)
        H[ti+1, :, :] = h_next
        # Save the value of the prediction in y (≈1 line)
        y_pred[ti, :, :] = yt_pred

    return H, y_pred
