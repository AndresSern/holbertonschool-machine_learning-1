#!/usr/bin/env python3
"""
builds a neural network with the Keras library
You are not allowed to use the Sequential class
"""

import tensorflow.keras as k


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    builds a neural network with the Keras library

    Args :
        *nx is the number of input features to the network
        *layers is a list containing the number of nodes
            in each layer of the network
        *activations is a list containing the activation
            functions used for each layer of the network
        *lambtha is the L2 regularization parameter
        *keep_prob is the probability that a node will
            be kept for dropout
    Retuen:
        the keras model
    """
    x = k.Input(shape=(nx,))
    for i in range(len(layers)):
        if i == 0:
            y = (k.layers.Dense(layers[i], activation=activations[i],
                 kernel_regularizer=k.regularizers.l2(lambtha)))(x)
        else:
            y = (k.layers.Dense(layers[i], activation=activations[i],
                 kernel_regularizer=k.regularizers.l2(lambtha)))(y)

        if i != len(layers) - 1:
            y = (k.layers.Dropout(1 - keep_prob))(y)
    model = k.Model(x, y)
    return model
