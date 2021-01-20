#!/usr/bin/env python3
"""
calculates the cost of a neural network with L2 regularization:
"""

import tensorflow as tf


def l2_reg_cost(cost):
    '''
    Args:
        *cost is a tensor containing the cost of
         the network without L2 regularization
        *Returns: a tensor containing the cost of
        the network accounting for L2 regularization
    '''
    a = cost + tf.losses.get_regularization_loss(scope=None)
    b = tf.compat.v1.norm(a, ord='euclidean', axis=None, keepdims=None, name=None, keep_dims=None)
    return b
