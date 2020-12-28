#!/usr/bin/env python3
"""
class NeuralNetwork that defines a neural
network with one hidden layer
performing binary classification
"""
import numpy as np
import matplotlib.pyplot as plt


class NeuralNetwork:
    """
    class NeuralNetwork
    with one hidden layer
    performing binary classification
    """
    def __init__(self, nx, nodes):
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")
        elif nx <= 0:
            raise ValueError("nx must be a positive integer")
        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")
        elif nodes <= 0:
            raise ValueError("nodes must be a positive integer")
        self.__W1 = np.random.normal(size=(nodes, nx))
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.normal(size=(1, nodes))
        self.__b2 = 0
        self.__A2 = 0

    @property
    def W1(self):
        """The weights vector for the hidden layer"""
        return self.__W1

    @property
    def b1(self):
        """The bias for the hidden layer. """
        return self.__b1

    @property
    def A1(self):
        """ The activated output for the hidden layer """
        return self.__A1

    @property
    def W2(self):
        """ The weights vector for the output neuron"""
        return self.__W2

    @property
    def b2(self):
        """The bias for the output neuron. """
        return self.__b2

    @property
    def A2(self):
        """ The activated output for the output neuron (prediction). """
        return self.__A2

    def forward_prop(self, X):
        """Calculates the forward propagation of the neural network
        X is a numpy.ndarray with shape (nx, m) that contains the input data
        nx is the number of input features to the neuron
        m is the number of examples
        """
        z1 = np.matmul(self.W1, X) + self.__b1
        Sigmoid_a1 = 1 / (1 + np.exp(-z1))
        self.__A1 = Sigmoid_a1
        z2 = np.matmul(self.W2, self.__A1) + self.__b2
        Sigmoid_a2 = 1 / (1 + np.exp(-z2))
        self.__A2 = Sigmoid_a2
        return self.__A1, self.__A2

    def cost(self, Y, A):
        """
        cost of the model using logistic regression
        """
        nx, m = Y.shape
        loss = - (Y * np.log(A) + (1 - Y) * np.log(1.0000001 - A))
        sumloss = np.sum(loss)
        cost = (1 / m) * sumloss
        return cost

    def evaluate(self, X, Y):
        """ evaluate The activated output
        for the output neuron (prediction)(A2)
        output 0 or 1
        """
        Sigmoid_a1, Sigmoid_a2 = self.forward_prop(X)
        pred_evalute = np.where(Sigmoid_a2 < 0.5, 0, 1)
        cost = self.cost(Y, Sigmoid_a2)
        return pred_evalute, cost

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """gradient_descent for  neural network with one hidden layer
        nx is the number of input features to the neuron
        m is the number of examples
        """
        ''' FOR THE OUTPUT hidden layer'''
        nx, m = Y.shape
        dz2 = A2 - Y
        dw2 = (1/m) * np.matmul(dz2, A1.T)
        db2 = (1/m) * np.sum(dz2, axis=1, keepdims=True)
        '''FOR THE one hidden layer
        dz1 = self.__W2.T * dz2 * A1 give the best accuracy
        '''
        dz1 = np.matmul(self.__W2.T, dz2) * (A1 * (1 - A1))
        dw1 = (1/m) * np.matmul(dz1, X.T)
        db1 = (1/m) * np.sum(dz1, axis=1, keepdims=True)
        self.__W2 -= (dw2 * alpha)
        self.__b2 -= (db2 * alpha)
        self.__W1 -= (dw1 * alpha)
        self.__b1 -= (db1 * alpha)
        return self.__W1, self.__b1, self.__W2, self.__b2

    def train(self, X, Y, iterations=5000, alpha=0.05, verbose=True,
              graph=True, step=100):
        """
        Returns the evaluation of the training data after
        iterations of training have occurred
        used:
        -forward_prop
        -gradient_descent
        - evaluate
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")
        elif iterations <= 0:
            raise ValueError("iterations must be a positive integer")
        elif not isinstance(alpha, float):
            raise TypeError("alpha must be a float")
        if alpha <= 0:
            raise ValueError("alpha must be positive")
        if verbose or graph:
            if type(step) is not int:
                raise TypeError("step must be an integer")
            if step < 0 or step > iterations:
                raise ValueError("step must be positive and <= iterations")
        xValue = []
        yValue = []
        for i in range(iterations+1):
            self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A1, self.__A2, alpha)
            cost = self.cost(Y, self.__A2)
            if verbose:
                if (i < 1 or i % step == 0):
                    print("Cost after {} iterations: {}".format(i, cost))
                    xValue.append(i+step)
                    yValue.append(cost)
        if graph is True:
            plt.title('Training Cost')
            plt.ylabel('cost')
            plt.xlabel('iteration')
            plt.plot(xValue, yValue)
            plt.show()
        return self.evaluate(X, Y)[0], cost
