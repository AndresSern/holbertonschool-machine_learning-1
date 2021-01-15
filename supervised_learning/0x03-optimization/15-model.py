#!/usr/bin/env python3
""" adam train fn """

import numpy as np
import tensorflow as tf


def create_placeholders(nx, classes):
    """ tf.placeholder"""
    x = tf.placeholder(tf.float32, shape=[None, nx], name="x")
    y = tf.placeholder(tf.float32, shape=[None, classes], name="y")
    return x, y


def create_layer(prev, n, activation):
    """ create layer"""
    kernel = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")
    layer = tf.layers.Dense(units=n, activation=activation,
                            kernel_initializer=kernel, name="layer",
                            )
    return layer(prev)


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


def forward_prop(x, layer_sizes=[], activations=[]):
    """forward propagation graph for the neural network"""
    for i, j in zip(layer_sizes, activations):
        if j is None:
            y = create_layer(x, i, j)
            x = y
        else:
            y = create_batch_norm_layer(x, i, j)
            x = y
    return y


def calculate_loss(y, y_pred):
    """calculate_loss"""
    loss = tf.losses.softmax_cross_entropy(y, y_pred)
    return loss


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """ ts adam """
    optimizer = tf.train.AdamOptimizer(
        learning_rate=alpha, beta1=beta1, beta2=beta2,
        epsilon=epsilon, use_locking=False, name='Adam')
    return optimizer.minimize(loss)


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """creates a learning rate decay
    operation in tensorflow using inverse time decay
    """
    inv = tf.train.inverse_time_decay(
        alpha,
        global_step,
        decay_step,
        decay_rate,
        staircase=True,
        name=None)
    return inv


def shuffle_data(X, Y):
    """
    Shuffles the data points in two matrices the same way
    """
    i = np.random.permutation(np.arange(X.shape[0]))
    return X[i], Y[i]


def calculate_accuracy(y, y_pred):
    """ calculate accuracy"""
    correct_prediction = tf.equal(tf.argmax(y_pred, axis=1),
                                  tf.argmax(y, axis=1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float32"))
    return accuracy


def split_data_batch(data, batch_size=32):
    """
    split_data_batch
    data = batch * batch_size

    """
    batches = []
    for i in range(batch_size, data.shape[0]+batch_size, batch_size):
        part = data[i-batch_size:i]
        batches.append(part)
    return(batches)


def model(Data_train, Data_valid, layers, activations,
          alpha=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8, decay_rate=1,
          batch_size=32, epochs=5, save_path='/tmp/model.ckpt'):
    """ my model"""

    X_train = Data_train[0]
    Y_train = Data_train[1]
    X_valid = Data_valid[0]
    Y_valid = Data_valid[1]

    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    y_pred = forward_prop(x, layers, activations)
    loss = calculate_loss(y, y_pred)
    accuracy = calculate_accuracy(y, y_pred)
    global_step = tf.Variable(0, trainable=False)
    decay = learning_rate_decay(alpha, decay_rate, global_step, 1)

    train_op = create_Adam_op(loss, decay, beta1, beta2, epsilon)
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    tf.add_to_collection('y_pred', y_pred)
    tf.add_to_collection('loss', loss)
    tf.add_to_collection('accuracy', accuracy)
    tf.add_to_collection('train_op', train_op)
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(init)
        for epoche in range(epochs+1):
            X_shuffled_train, Y_shuffled_train = shuffle_data(X_train, Y_train)
            train_acc, train_loss = sess.run((accuracy, loss),
                                             {x: X_train, y: Y_train})
            valid_acc, valid_loss = sess.run((accuracy, loss),
                                             {x: X_valid, y: Y_valid})

            print("After {} epochs:".format(epoche))
            print("\tTraining Cost: {}".format(train_loss))
            print("\tTraining Accuracy: {}".format(train_acc))
            print("\tValidation Cost: {}".format(valid_loss))
            print("\tValidation Accuracy: {}".format(
                valid_acc))

            if epoche < epochs:
                mini_batch_X_t = split_data_batch(X_shuffled_train,
                                                  batch_size)
                mini_batch_Y_t = split_data_batch(Y_shuffled_train,
                                                  batch_size)
                batchline = len(mini_batch_X_t)
                for step in range(0, batchline):

                    sess.run(train_op, {
                        x: mini_batch_X_t[step],
                        y: mini_batch_Y_t[step]
                    })

                    train_loss, train_acc = sess.run((loss, accuracy), {
                        x: mini_batch_X_t[step],
                        y: mini_batch_Y_t[step]
                    })
                    if((step + 1) % 100 == 0 and (step + 1) > 0):
                        print("\tStep {}:".format(step + 1))
                        print("\t\tCost: {}".format(train_loss))
                        print("\t\tAccuracy: {}".format(train_acc))
            sess.run(tf.assign(global_step, global_step + 1))
        return saver.save(sess, save_path)
