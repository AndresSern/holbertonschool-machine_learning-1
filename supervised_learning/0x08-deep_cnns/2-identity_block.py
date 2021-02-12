#!/usr/bin/env python3
"""
builds an identity block as described in Deep Residual
Learning for Image Recognition (2015):
"""
import tensorflow.keras as K


def identity_block(A_prev, filters):
    """
    A_prev is the output from the previous layer
    filters is a tuple or list containing F11, F3, F12, respectively:
    F11 is the number of filters in the first 1x1 convolution
    F3 is the number of filters in the 3x3 convolution
    F12 is the number of filters in the second 1x1 convolution
    """
    kernel = K.initializers.he_normal()

    F11, F2, F12 = filters

    X = K.layers.Conv2D(filters=F11, kernel_size=(1, 1), strides=(1, 1),
                        padding='same', kernel_initializer=kernel)(A_prev)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    X = K.layers.Conv2D(filters=F2, kernel_size=(3, 3), strides=(1, 1),
                        padding='same', kernel_initializer=kernel)(X)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    X = K.layers.Conv2D(filters=F12, kernel_size=(1, 1), strides=(1, 1),
                        padding='same', kernel_initializer=kernel)(X)
    X = K.layers.BatchNormalization(axis=3)(X)
    """ A_prev is the shortcut"""
    X = K.layers.Add()([X, A_prev])
    X = K.layers.Activation('relu')(X)
    return X
