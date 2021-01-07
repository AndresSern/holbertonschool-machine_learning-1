#!/usr/bin/env python3
""" calculate accuracy"""

import tensorflow as tf


def calculate_accuracy(y, y_pred):
    """ calculate accuracy"""
    correct_prediction = tf.equal(tf.argmax(y_pred, axis=1),
                                  tf.argmax(y, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float32"))
    return accuracy
