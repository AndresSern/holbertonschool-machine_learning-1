#!/usr/bin/env python3
''' evaluate fn'''

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf


def evaluate(X, Y, save_path):
    ''' evaluate'''
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        saver = tf.train.import_meta_graph('{}'.format(save_path))
        saver.restore(sess, '{}'.format(save_path))
        x = tf.get_collection('x', scope=None)
    print(x)
