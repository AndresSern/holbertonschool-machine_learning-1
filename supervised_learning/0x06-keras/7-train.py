#!/usr/bin/env python3
"""
trains a model using mini-batch gradient descent
and also analyze validaiton data
and  also train the model using early stopping
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow._api.v1.keras.utils import to_categorical


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False, patience=0,
                learning_rate_decay=False, alpha=0.1, decay_rate=1,
                verbose=True, shuffle=False):
    """
    Args:
        *network is the model to train
        *data is a numpy.ndarray of shape (m, nx)
            containing the input data
        *labels is a one-hot numpy.ndarray of shape(m, classes)
            containing the labels of data
        *batch_size is the size of the batch used for
            mini-batch gradient descent
        *epochs is the number of passes through data for
            mini-batch gradient descent
        *verbose is a boolean that determines if output should
            be printed during training
        *shuffle is a boolean that determines whether to shuffle
            the batches every epoch. Normally, it is a
        *validation_data is the data to validate the model with, if not None

    Returns: the History object generated after training the model
    """
    if validation_data and early_stopping:
        if learning_rate_decay:
            lr_schedule = keras.optimizers.schedules.ExponentialDecay(
                initial_learning_rate=alpha,
                decay_steps=epochs,
                decay_rate=decay_rate)
            optimizer = keras.optimizers.SGD(learning_rate=lr_schedule)
        callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=patience,verbose=verbose)
        history = network.fit(x=data, y=labels, batch_size=batch_size,
                            epochs=epochs, verbose=verbose,
                            validation_data=validation_data, shuffle=shuffle, callbacks=[callback])
    else:
        history = network.fit(x=data, y=labels, batch_size=batch_size,
                        epochs=epochs, verbose=verbose,
                        validation_data=validation_data, shuffle=shuffle)
    return history