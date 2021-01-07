#!/usr/bin/env python3
""" calculate accuracy"""

import tensorflow as tf
def calculate_accuracy(y, y_pred):
    """ calculate accuracy"""
    correct_prediction = tf.equal(y_pred, y)
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    return accuracy