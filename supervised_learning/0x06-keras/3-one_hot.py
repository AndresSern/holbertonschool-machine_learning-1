#!/usr/bin/env python3
"""
converts a label vector into a one-hot matrix
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow._api.v1.keras.utils import to_categorical


def one_hot(labels, classes=None):
    """ keras one hot encoding"""
    encoded = to_categorical(labels)
    return encoded
