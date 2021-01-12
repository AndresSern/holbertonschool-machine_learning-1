#!/usr/bin/env python3
""" MomentumOptimizer """
import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """MomentumOptimizer """
    optimizer = tf.train.MomentumOptimizer(alpha, beta1)
    return optimizer.minimize(loss)
