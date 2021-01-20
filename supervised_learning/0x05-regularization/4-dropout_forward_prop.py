#!/usr/bin/env python3
'''
conducts forward propagation using Dropout
'''
import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):

    """
    Args:

        *X is a numpy.ndarray of shape (nx, m) containing
            the input data for the network
        *nx is the number of input features
        *m is the number of data points
        *weights is a dictionary of the weights and biases
            of the neural network
        *L the number of layers in the network
        *keep_prob is the probability that a node will be kept
        *All layers except the last should use the tanh activation function
        *The last layer should use the softmax activation function

    Returns: a dictionary containing the outputs of each layer
        and the dropout mask used on each layer (see example for format)

    """
    cache = {}
    cache["A0"] = X
    for i in range(L):
        w = weights["W"+str(i+1)]
        b = weights["b" + str(i+1)]
        z = np.matmul(w, cache["A"+str(i)]) + b
        if(i != L - 1):
            tanh = (2 / (1 + np.exp(-2 * z))) - 1
            '''
            we can use also:
            d = np.random.rand(tanh.shape[0],tanh.shape[1])
            d = d < keep_prob
            a = np.multiply(tanh , d)
            cache["D"+str(i+1)] = np.multiply(d, 1)
            '''
            d = np.random.binomial(1, keep_prob, size=tanh.shape)
            a = tanh * d
            a /= keep_prob
            cache["A"+str(i+1)] = a
            cache["D"+str(i+1)] = d

        else:
            """ softmax for the output layer
            Softmax function returns probabilities sum to 1
            """
            t = np.exp(z - np.max(z))
            softmax = t / t.sum(axis=0)
            cache["A"+str(i+1)] = softmax
    return cache
