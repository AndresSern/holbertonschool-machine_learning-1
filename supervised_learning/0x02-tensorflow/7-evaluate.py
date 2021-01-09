#!/usr/bin/env python3
''' evaluate fn'''

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def evaluate(X, Y, save_path):
    ''' evaluate FN'''
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        saver = tf.train.import_meta_graph('{}.meta'.format(save_path))
        saver.restore(sess, '{}'.format(save_path))
        ax = tf.get_collection('x', scope=None)
        ay = tf.get_collection('y', scope=None)
        ay_pred = tf.get_collection('y_pred', scope=None)
        aloss = tf.get_collection('loss', scope=None)
        aaccuracy = tf.get_collection('accuracy', scope=None)

        y_pred = ay_pred[0]
        loss = aloss[0]
        accuracy = aaccuracy[0]
        x = ax[0]
        y = ay[0]

        y_pred = sess.run(y_pred, {x: X, y: Y})
        accuracy = sess.run(aaccuracy, {x: X, y: Y})
        loss = sess.run(aloss, {x: X, y: Y})
    return(y_pred, accuracy, loss)
