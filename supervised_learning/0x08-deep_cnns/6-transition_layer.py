#!/usr/bin/env python3
"""
builds a transition layer as described in
Densely Connected Convolutional Networks
"""

import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """
    X is the output from the previous layer
    nb_filters is an integer representing the number of filters in X
    compression is the compression factor for the transition layer

    Returns: The output of the transition layer and the number of
    filters within the output, respectively

    """
    kernel = K.initializers.he_normal()
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)
    """
    compression: calculated as 1 - reduction. Reduces the number
    of feature maps in the transition block.
    """
    a = nb_filters * compression
    X = K.layers.Conv2D(filters=(int(a)),
                        kernel_size=(1, 1),
                        padding="same",
                        kernel_initializer=kernel)(X)
    X = K.layers.AveragePooling2D()(X)
    return X, int(a)
