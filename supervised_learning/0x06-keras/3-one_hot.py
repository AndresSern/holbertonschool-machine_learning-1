#!/usr/bin/env python3
"""
converts a label vector into a one-hot matrix
"""

import tensorflow.keras as k


def one_hot(labels, classes=None):
    """ keras one hot encoding"""
    encoded = k.utils.to_categorical(labels, classes)
    return encoded
