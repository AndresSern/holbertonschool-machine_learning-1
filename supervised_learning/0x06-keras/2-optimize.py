#!/usr/bin/env python3
"""
up Adam optimization for a keras model with categorical
crossentropy loss and accuracy metrics:
"""

import tensorflow.keras as k


def optimize_model(network, alpha, beta1, beta2):
    "adam optimization"
    opt = k.optimizers.Adam(lr=alpha, beta_1=beta1, beta_2=beta2)
    network.compile(loss='categorical_crossentropy',
                    metrics=['accuracy'], optimizer=opt)
    return None
