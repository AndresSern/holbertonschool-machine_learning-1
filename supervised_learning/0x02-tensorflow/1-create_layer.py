#!/usr/bin/env python3
""" create layer"""

import tensorflow as tf


def create_layer(prev, n, activation):
    """ create layer"""
    kernel = tf.contrib.layers.variance_scaling_initializer(factor=2.0,
                                                            mode="FAN_AVG")
    layer = tf.layers.dense(prev, units=n, activation=activation,
                            kernel_initializer=kernel, name="layer",
                            reuse=tf.AUTO_REUSE)
    return layer
