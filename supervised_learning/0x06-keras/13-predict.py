#!/usr/bin/env python3
"""
that makes a prediction using a neural network:

    *network is the network model to make
        the prediction with
    *data is the input data to make the prediction with
    *verbose is a boolean that determines if output
        should be printed during the prediction process
Returns: the prediction for the data

"""

import tensorflow.keras as K


def predict(network, data, verbose=False):
    """ makes a prediction"""
    pred = network.predict(data, verbose=verbose)
    return pred
