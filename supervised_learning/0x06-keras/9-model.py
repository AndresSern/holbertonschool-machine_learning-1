#!/usr/bin/env python3


"""
saves an entire model
and loads an entire model
Args:
    *network is the model
    *filename is the path of the file that
        the model should be saved to
"""
import tensorflow.keras as K


def save_model(network, filename):
    """ saves an entire model"""
    network.save(filename)
    return None


def load_model(filename):
    """ loads an entire model"""
    network = K.models.load_model(filename)
    return network
