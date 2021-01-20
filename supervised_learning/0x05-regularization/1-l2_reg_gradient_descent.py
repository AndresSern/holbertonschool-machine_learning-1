#!/usr/bin/env python3
"""
Updates the weights and biases of a neural
network using gradient descent with L2 regularization
"""
import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    '''
    Args:
        *Y is a one-hot numpy.ndarray of shape (classes, m)
            that contains the correct labels for the data
            *classes is the number of classes
            *m is the number of data points

        *weights is a dictionary of the weights and biases of
            the neural network

        *cache is a dictionary of the outputs of each layer of
            the neural network

        *alpha is the learning rate

        *lambtha is the L2 regularization parameter

        *L is the number of layers of the network

        *The neural network uses tanh activations on each layer
        except the last, which uses a softmax activation

        *The weights and biases of the network should be updated in place
    '''
    v_weights = weights.copy()
    classes, m = Y.shape
    dz = cache["A"+str(L)] - Y
    for i in range(L, 0, -1):
        A_i = "A"+str(i-1)
        wi = "W"+str(i)
        bi = "b"+str(i)
        dw = ((1/m) * np.matmul(
            dz, cache["A"+str(i-1)].T)) + ((lambtha/m) * weights[wi])
        db = (1/m) * np.sum(dz, axis=1, keepdims=True)
        weights[wi] = weights[wi] - (dw * alpha)
        weights[bi] = weights[bi] - (db * alpha)

        dz = np.matmul(v_weights[wi].T, dz) * (
            1 - np.power(cache[A_i], 2))
    return weights
