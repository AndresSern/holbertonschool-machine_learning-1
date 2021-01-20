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
    reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    reg_constant = 0.01
    cst = cost + reg_constant * sum(reg_losses)
    return cst
