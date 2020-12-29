#!/usr/bin/env python3
"""
 DeepNeuralNetwork that defines a deep
 neural network performing binary classification
"""
import numpy as np


class DeepNeuralNetwork:
    """
    class DeepNeuralNetwork
    nx is the number of input features
    layers is a list representing the number
    of nodes in each layer of the network
    L: The number of layers in the neural network
    """
    def __init__(self, nx, layers):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        elif nx <= 0:
            raise ValueError("nx must be a positive integer")
        if not isinstance(layers, list) or not layers:
            raise TypeError("layers must be a list of positive integers")
        self.L = len(layers)
        self.cache = {}
        self.weights = {}
        for i in range(len(layers)):
            if not isinstance(layers[i], int) or layers[i] <= 0:
                raise TypeError("must be a list of positive integers")
            if i == 0:
                self.weights['b' + str(i + 1)] = np.zeros((layers[0], 1))
                self.weights['W' + str(i + 1)
                             ] = np.random.normal(size=(layers[i], nx)
                                                  ) * np.sqrt(2/nx)
                
            else:
                self.weights['b' + str(i + 1)] = np.zeros((layers[i], 1))
                self.weights['W' + str(i + 1)
                             ] = np.random.normal(size=(layers[i], layers[i-1])
                                                  )*np.sqrt(
                                                            2/layers[i-1])
                
