#!/usr/bin/env python3
import tensorflow as tf


def lenet5(x, y):
    kernel = tf.contrib.layers.variance_scaling_initializer()
    """Layer 1: Convolutional. Input = 32x32x3. Output = 32x32x6."""
    conv1 = tf.layers.conv2d(x, kernel_size=(5, 5),
                             strides=1, padding='SAME',
                             kernel_initializer=kernel,
                             filters=6, activation='relu')

    """Pooling. Input = 32x32x6. Output = 14x14x6"""
    pool_1 = tf.layers.max_pooling2d(conv1, pool_size=(2, 2),
                                     strides=2, padding='SAME')

    """Layer 2: Convolutional. Output = 10x10x16."""
    conv2 = tf.layers.conv2d(pool_1, kernel_size=(5, 5),
                             strides=1, padding='VALID',
                             kernel_initializer=kernel,
                             filters=16, activation='relu')

    """Pooling. Input = 10x10x16. Output = 5x5x16."""
    pool_2 = tf.layers.max_pooling2d(conv2, pool_size=(2, 2),
                                     strides=2, padding='VALID')

    """ Flatten. Input = 5x5x16. Output = 400."""
    fc0 = tf.contrib.layers.flatten(pool_2)

    """Layer 3: Fully Connected. Input = 400. Output = 120."""
    fc1 = tf.layers.dense(inputs=fc0, units=120, activation='relu',
                          kernel_initializer=kernel)

    """Layer 4: Fully Connected. Input = 120. Output = 84."""
    fc2 = tf.layers.dense(inputs=fc1, units=84, activation='relu',
                          kernel_initializer=kernel)

    """Layer 5: Fully Connected. Input = 84. Output = 10."""
    logits = tf.layers.dense(inputs=fc2, units=10,
                             kernel_initializer=kernel)

    softmax = tf.nn.softmax(logits)
    loss = tf.losses.softmax_cross_entropy(y, logits)

    optimizer = tf.train.AdamOptimizer().minimize(loss)
    correct_prediction = tf.equal(tf.argmax(logits, axis=1),
                                  tf.argmax(y, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float32"))

    return softmax, optimizer, loss, accuracy
