#!/usr/bin/env python3
""" forward_prop"""

import tensorflow as tf


def forward_prop(x, layer_sizes=[], activations=[]):
    """ forward_prop  n time tf.layers.Dense
        to get the output prediction Y
    """
    create_layer = __import__('1-create_layer').create_layer
    y = create_layer(x, layer_sizes[0], activations[0])
    for i in range(1, len(layer_sizes)):
        y = create_layer(y, layer_sizes[i], activations[i])
    return y
