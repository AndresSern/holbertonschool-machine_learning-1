#!/usr/bin/env python3
"""
trains a model using mini-batch gradient descent
and also analyze validaiton data
and  also train the model using early stopping
"""

import tensorflow.keras as k


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
    if validation_data:
        calbacks = []
        if learning_rate_decay:

            def decayed_learning_rate(step):
                return alpha / (1 + decay_rate * step)
            callback = k.callbacks.LearningRateScheduler(decayed_learning_rate,
                                                         verbose=1)
            calbacks.append(callback)
        if early_stopping:
            callback = k.callbacks.EarlyStopping(monitor='val_loss',
                                                 patience=patience,
                                                 verbose=verbose)
            calbacks.append(callback)
        history = network.fit(x=data, y=labels, batch_size=batch_size,
                              epochs=epochs, verbose=verbose,
                              validation_data=validation_data,
                              shuffle=shuffle, callbacks=calbacks)
    else:
        history = network.fit(x=data, y=labels, batch_size=batch_size,
                              pochs=epochs, verbose=verbose,
                              validation_data=validation_data,
                              shuffle=shuffle)
    return history
