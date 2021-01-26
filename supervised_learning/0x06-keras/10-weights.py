#!/usr/bin/env python3
"""
saves and loads a model’s weights:
args:
    *network is the model whose weights
        should be saved
    *filename is the path of the file that
        the weights should be saved to
    *save_format is the format in which
        the weights should be saved
"""

import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    """saves a model’s weights:"""
    network.save_weights(filename, save_format=save_format)
    return None


def load_weights(network, filename):
    """loads a model’s weights:"""
    network.load_weights(filename)
    return None
