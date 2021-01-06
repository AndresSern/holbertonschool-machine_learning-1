#!/usr/bin/env python3
""" create layer"""

import tensorflow as tf


def create_layer(prev, n, activation):
    """ create layer"""
    kernel = tf.contrib.layers.variance_scaling_initializer(factor=2.0,
                                                            mode="FAN_AVG")
    layer = tf.layers.dense(prev, units=n, activation=activation,
                            kernel_initializer=kernel, name="layer",
                            bias_regularizer=None, activity_regularizer=None,
                            kernel_constraint=None,
                            bias_constraint=True,trainable=None)
    return layer
