#!/usr/bin/env python3
"""
saves and loads model’s configuration in JSON format:
args:
    *network is the model whose weights
        should be saved
    *filename is the path of the file that
        the weights should be saved to

"""

import tensorflow.keras as K


def save_config(network, filename):
    """ saves  model’s configuration in JSON format:"""
    network = network.to_json()
    with open(filename, "w") as json_file:
        json_file.write(network)
    return None


def load_config(filename):
    """loads model’s configuration in JSON format:"""
    json_file = open(filename, 'r')
    loaded_model_json = json_file.read()
    a = K.models.model_from_json(loaded_model_json, custom_objects=None)
    return a
