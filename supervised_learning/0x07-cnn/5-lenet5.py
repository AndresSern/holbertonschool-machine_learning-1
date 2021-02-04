#!/usr/bin/env python3
"""
a modified version of the LeNet-5 architecture using keras
"""
import tensorflow.keras as K


def lenet5(X):

    """ initialize their kernels with the he_normal initialization method"""
    kernel = K.initializers.he_normal(seed=None)

    """Layer 1: Convolutional. Input = 28x28x1. Output = 28x28x6."""
    conv1 = K.layers.Conv2D(filters=6, kernel_size=5,
                            padding='same', activation='relu',
                            kernel_initializer=kernel)(X)

    """Pooling. Input = 28x28x6. Output = 14x14x6"""
    pool_1 = K.layers.MaxPooling2D(pool_size=2, strides=2)(conv1)

    """Layer 2: Convolutional. Output = 10x10x16."""
    conv2 = K.layers.Conv2D(filters=16, kernel_size=5,
                            padding='valid', activation='relu',
                            kernel_initializer=kernel)(pool_1)

    """Pooling. Input = 10x10x16. Output = 5x5x16."""
    pool_2 = K.layers.MaxPooling2D(pool_size=2, strides=2,
                                   input_shape=(10, 10, 16))(conv2)

    """ Flatten. Input = 5x5x16. Output = 400."""
    flat_1 = K.layers.Flatten()(pool_2)

    """Layer : Fully Connected. Input = 400. Output = 120."""
    layer_1 = K.layers.Dense(units=120, activation='relu',
                             kernel_initializer=kernel)(flat_1)

    """Layer : Fully Connected. Input = 120. Output = 84."""
    layer_2 = K.layers.Dense(units=84, activation='relu',
                             kernel_initializer=kernel)(layer_1)

    """Layer 5: Fully Connected. Input = 84. Output = 10."""
    layer_3 = K.layers.Dense(units=10, activation='softmax',
                             kernel_initializer=kernel)(layer_2)

    """
    a K.Model compiled to use Adam optimization
    (with default hyperparameters) and accuracy metrics
    """
    model = K.models.Model(inputs=X, outputs=layer_3)
    adam = K.optimizers.Adam()
    model.compile(optimizer=adam, loss='categorical_crossentropy',
                  metrics=['accuracy'])

    return model
