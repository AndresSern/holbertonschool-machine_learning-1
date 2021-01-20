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
    regularizer = tf.contrib.layers.l2_regularizer(scale=0.1)
    reg_variables = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    reg_term = tf.contrib.layers.apply_regularization(regularizer,
                                                      reg_variables)
    cost += reg_term
    return cost
