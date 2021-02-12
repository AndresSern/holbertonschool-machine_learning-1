#!/usr/bin/env python3
"""
builds the ResNet-50 architecture as described
in Deep Residual Learning for Image Recognition (2015):
"""

import tensorflow.keras as K
identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """
    input data will have shape (224, 224, 3)
    he normal initialization
    """
    kernel = K.initializers.he_normal()
    X_input = K.Input(shape=(224, 224, 3))
    '''K.layers.ZeroPadding2D((3, 3))(X_input)'''
    """ 7x7 ,64, stride = 2"""
    X = K.layers.Conv2D(64, (7, 7), strides=(2, 2),
                        kernel_initializer=kernel, padding='same')(X_input)
    X = K.layers.BatchNormalization()(X)
    X = K.layers.Activation('relu')(X)
    """ 3x3 maxpool stride = 2"""
    X = K.layers.MaxPooling2D((3, 3), strides=(2, 2), padding='same')(X)
    """ conv2_x * 3"""
    X = projection_block(X, [64, 64, 256], s=1)
    X = identity_block(X, [64, 64, 256])
    X = identity_block(X, [64, 64, 256])

    """ conv2_x * 4"""
    X = projection_block(X, [128, 128, 512])
    X = identity_block(X, [128, 128, 512])
    X = identity_block(X, [128, 128, 512])
    X = identity_block(X, [128, 128, 512])
    """ conv2_x * 6"""
    X = projection_block(X, [256, 256, 1024])
    X = identity_block(X, [256, 256, 1024])
    X = identity_block(X,  [256, 256, 1024])
    X = identity_block(X,  [256, 256, 1024])
    X = identity_block(X,  [256, 256, 1024])
    X = identity_block(X,  [256, 256, 1024])
    """ conv2_x * 3"""
    X = projection_block(X, [512, 512, 2048])
    X = identity_block(X,  [512, 512, 2048])
    X = identity_block(X,  [512, 512, 2048])
    """ AveragePooling2D '1000-d fc'"""
    X = K.layers.AveragePooling2D(pool_size=(7, 7), padding='same')(X)

    X = K.layers.Dense(1000, activation='softmax',
                       kernel_initializer=kernel)(X)
    """ model"""
    model = K.models.Model(inputs=X_input, outputs=X)

    return model
