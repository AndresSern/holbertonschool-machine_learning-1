#!/usr/bin/env python3
""" learning rate decay"""


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    updates the learning rate using
    inverse time decay in numpy
    """
    a0 = alpha
    a = global_step / decay_step
    a = int(a)
    alpha = (a0 / (1 + (decay_rate * a)))
    return alpha
