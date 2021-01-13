#!/usr/bin/env python3
"""
*prev is the activated output of the previous layer
*n is the number of nodes in the layer to be created
*activation is the activation function that
should be used on the output of the layer
*gamma and beta, initialized as vectors of 1 and 0 respectively
*epsilon of 1e-8
Returns: a tensor of the activated output for the layer
"""
import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """batch normalization layer for a neural network in tensorflow"""
    kernel = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, activation=None,
                            kernel_initializer=kernel, name="layer",
                            )

    z = layer(prev)

    batch_mean1, batch_var1 = tf.nn.moments(z, [0])
    gamma = tf.Variable(tf.ones([n]))
    beta = tf.Variable(tf.zeros([n]))
    BN = tf.nn.batch_normalization(z, mean=batch_mean1, variance=batch_var1,
                                   offset=beta, scale=gamma,
                                   variance_epsilon=1e-8
                                   )
    return activation(BN)
