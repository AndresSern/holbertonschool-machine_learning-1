#!/usr/bin/env python3
"""
builds a projection block as described in
Deep Residual Learning for Image Recognition (2015):
"""
import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """
    A_prev___is the output from the previous layer
    filters___is a tuple or list containing F11, F3, F12, respectively:
    F11___ is the number of filters in the first 1x1 convolution
    F3___ is the number of filters in the 3x3 convolution
    F12___ is the number of filters in the second 1x1 convolution
    s___ is the stride of the first convolution in both
        the main path and the shortcut connection

    Returns: the activated output of the projection block
    """
    kernel = K.initializers.he_normal()

    F11, F2, F12 = filters

    X = K.layers.Conv2D(filters=F11, kernel_size=(1, 1), strides=(s, s),
                        padding='valid', kernel_initializer=kernel)(A_prev)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    X = K.layers.Conv2D(filters=F2, kernel_size=(3, 3), strides=(1, 1),
                        padding='same', kernel_initializer=kernel)(X)
    X = K.layers.BatchNormalization(axis=3)(X)
    X = K.layers.Activation('relu')(X)

    X = K.layers.Conv2D(filters=F12, kernel_size=(1, 1), strides=(1, 1),
                        padding='valid', kernel_initializer=kernel)(X)
    X = K.layers.BatchNormalization(axis=3)(X)

    shortcut = K.layers.Conv2D(F12, (1, 1), strides=(s, s),
                               kernel_initializer=kernel,
                               )(A_prev)
    shortcut = K.layers.BatchNormalization(axis=3,)(shortcut)

    X = K.layers.Add()([X, shortcut])
    X = K.layers.Activation('relu')(X)
    return X
