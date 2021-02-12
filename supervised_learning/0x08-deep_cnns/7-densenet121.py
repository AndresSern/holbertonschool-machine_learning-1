#!/usr/bin/env python3
"""
that builds the DenseNet-121 architecture as described
in Densely Connected Convolutional Networks:
"""
import tensorflow.keras as K
dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """
    args
    growth_rate is the growth rate
    compression is the compression factor

    Returns: the keras model
    """
    X_input = K.Input(shape=(224, 224, 3))
    kernel = K.initializers.he_normal()

    X = K.layers.BatchNormalization(axis=3, name='bn_conv1')(X_input)
    X = K.layers.Activation('relu')(X)
    """
    nb_filter: initial number of filters. Default -1 indicates initial number
    of filters is 2 * growth_rate
    """
    filters = 0
    if filters <= 0:
        filters = 2 * growth_rate
    X = K.layers.Conv2D(filters, (7, 7), strides=(2, 2),
                        kernel_initializer=kernel)(X)

    X = K.layers.MaxPooling2D(3, strides=2)(X)

    X, filters = dense_block(X, nb_filters=filters,
                             growth_rate=growth_rate, layers=6)
    X, filters = transition_layer(X, filters, compression)
    X, filters = dense_block(X, filters, growth_rate, 12)
    X, filters = transition_layer(X, filters, compression)
    X, filters = dense_block(X, filters, growth_rate, 24)
    X, filters = transition_layer(X, filters, compression)
    X, filters = dense_block(X, filters, growth_rate, 16)
    X = K.layers.AveragePooling2D(pool_size=6)(X)
    output = K.layers.Dense(units=1000, activation='softmax',
                            kernel_initializer=kernel)(X)
    return K.models.Model(inputs=X_input, outputs=output)
