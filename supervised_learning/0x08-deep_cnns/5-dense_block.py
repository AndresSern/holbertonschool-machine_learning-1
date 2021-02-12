#!/usr/bin/env python3
"""
that builds a dense block as described in Densely
Connected Convolutional Networks:
"""

import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """
    ARGs:

    X is the output from the previous layer
    nb_filters is an integer representing the number of filters in X
    growth_rate is the growth rate for the dense block
    layers is the number of layers in the dense block

    Returns: The concatenated output of each layer within the
    Dense Block and the number of filters within the concatenated
    outputs, respectively
    """
    shortcut = X
    kernel = K.initializers.he_normal()
    for i in range(layers):
        """the bottleneck layers used for DenseNet-B"""
        """
        DenseNet-B: 1x1 conv bottleneck before 3x3 conv
        """

        """ bottleneck convolution block"""
        X = K.layers.BatchNormalization(axis=3)(shortcut)
        X = K.layers.Activation('relu')(X)
        inter_channel = growth_rate * 4
        X = K.layers.Conv2D(filters=(inter_channel),
                            kernel_size=(1, 1),
                            padding="same",
                            kernel_initializer=kernel)(X)
        """ end of bottleneck convolution block """

        X = K.layers.BatchNormalization(axis=3)(X)
        X = K.layers.Activation('relu')(X)
        X = K.layers.Conv2D(filters=growth_rate,
                            kernel_size=(3, 3),
                            padding="same",
                            kernel_initializer=kernel)(X)

        shortcut = K.layers.Concatenate()([shortcut, X])
        nb_filters += growth_rate
    return shortcut, nb_filters
