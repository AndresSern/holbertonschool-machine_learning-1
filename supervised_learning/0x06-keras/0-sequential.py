#!/usr/bin/env python3
"""
that builds a neural network with the Keras library
You are not allowed to use the Input class
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
    model = k.Sequential()
    for i in range(len(layers)):
        if i == 0:
            d = nx
        else:
            d = layers[i - 1]

        model.add(k.layers.Dense(layers[i], input_dim=d,
                  activation=activations[i],
                  kernel_regularizer=k.regularizers.l2(lambtha)))
        if i != len(layers) - 1:
            model.add(k.layers.Dropout(1-keep_prob))
    return model
