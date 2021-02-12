#!/usr/bin/env python3
"""
builds an inception block as described in Going Deeper
with Convolutions (2014):
"""
import tensorflow.keras as K


def inception_block(A_prev, filters):
    """
    A_prev is the output from the previous layer
    filters is a tuple or list  F1, F3R, F3,F5R, F5, FPP, respectively:
    F1 is the number of filters in the 1x1 convolution

    F3R is the number of filters in the 1x1 convolution before
        the 3x3 convolution

    F3 is the number of filters in the 3x3 convolution

    F5R is the number of filters in the 1x1 convolution before
        the 5x5 convolution

    F5 is the number of filters in the 5x5 convolution
    FPP is the number of filters in the 1x1 convolution after the max pooling
    All convolutions inside the inception block  use activation (ReLU)
    Returns: the concatenated output of the inception block
    """
    kernel = K.initializers.he_normal()
    tower_0 = K.layers.Conv2D(filters[0], kernel_size=(1, 1), padding='same',
                              activation='relu',
                              kernel_initializer=kernel)(A_prev)

    tower_1 = K.layers.Conv2D(filters[1], (1, 1), padding='same',
                              activation='relu',
                              kernel_initializer=kernel)(A_prev)
    tower_1 = K.layers.Conv2D(filters[2], (3, 3), padding='same',
                              activation='relu',
                              kernel_initializer=kernel)(tower_1)

    tower_2 = K.layers.Conv2D(filters[3], (1, 1), padding='same',
                              activation='relu',
                              kernel_initializer=kernel)(A_prev)

    tower_2 = K.layers.Conv2D(filters[4], (5, 5), padding='same',
                              activation='relu',
                              kernel_initializer=kernel)(tower_2)

    tower_3 = K.layers.MaxPooling2D(strides=(1, 1), padding='same')(A_prev)
    tower_3 = K.layers.Conv2D(filters[5], (1, 1), padding='same',
                              activation='relu',
                              kernel_initializer=kernel)(tower_3)

    output = K.layers.concatenate([tower_0, tower_1,
                                  tower_2, tower_3], axis=3)
    return output
