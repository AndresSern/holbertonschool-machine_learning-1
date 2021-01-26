#!/usr/bin/env python3
"""
builds a neural network with the Keras library
You are not allowed to use the Sequential class
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow._api.v1.keras import Sequential
from tensorflow._api.v1.keras.layers import Dense
from tensorflow._api.v1.keras.layers import Dropout
from tensorflow._api.v1.keras import regularizers


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
    x = tf.keras.Input(shape=(nx,))
    for i in range(len(layers)):
        if i == 0:
            y = (Dense(layers[i], activation=activations[i],
                       kernel_regularizer=regularizers.l2(lambtha)))(x)
        else:
            y = (Dense(layers[i], activation=activations[i],
                 kernel_regularizer=regularizers.l2(lambtha)))(y)

        if i != len(layers) - 1:
            y = (Dropout(1 - keep_prob))(y)
    model = tf.keras.Model(x, y)
    return model
