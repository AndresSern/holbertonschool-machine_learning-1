#!/usr/bin/env python3
""" forward propagation for a deep RNN"""

import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """
    ARGS:
        -rnn_cells: {list of RNNCell instances}  of length l
            -l is the number of layers
        -X :{ numpy.ndarray} of shape (t, m, i) :is the data to be used
            -t is the maximum number of time steps
            -m is the batch size
            -i is the dimensionality of the data
        -h_0 : {numpy.ndarray} shape (l, m, h): the initial hidden state
            -h is the dimensionality of the hidden state
    Returns: H, Y
        -H :{numpy.ndarray} : containing all of the hidden states
        -Y :{numpy.ndarray} : containing all of the outputs
    """
    H = []
    Y = []
    H.append(h_0)
    for j in range(X.shape[0]):
        ht = []
        h = X[j]
        for i in range(len(rnn_cells)):
            h, y = rnn_cells[i].forward(H[j][i], h)
            ht.append(h)
        H.append(ht)
        Y.append(y)
    H = np.array(H)
    Y = np.array(Y)
    return H, Y
