#!/usr/bin/env python3
"""
up Adam optimization for a keras model with categorical
crossentropy loss and accuracy metrics:
"""

import tensorflow as tf
from tensorflow import keras
#from tensorflow._api.v1.keras.metrics import Accuracy


def optimize_model(network, alpha, beta1, beta2):
    "adam optimization"
    opt = keras.optimizers.Adam(lr=alpha, beta_1=beta1, beta_2=beta2)
    network.compile(loss='categorical_crossentropy',
                    metrics=['accuracy'],optimizer=opt)
    return None
