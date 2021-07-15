#!/usr/bin/env python3
"""
computes the Monte-Carlo policy gradient based
on a state and a weight matrix.
"""
import numpy as np


def softmax_grad(probs):
    """Vectorized softmax"""
    s = probs.reshape(-1, 1)
    return np.diagflat(s) - np.dot(s, s.T)


def softmax(vector):
    """softmax function"""
    e = np.exp(vector)
    return e / e.sum()


def policy(state, weight):
    """
    maps state to action parameterized by w
    """
    π = np.dot(state, weight)
    return softmax(π)


def policy_gradient(state, weight):
    """
    function that computes the Monte-Carlo policy gradient based
    on a state and a weight matrix.
    ARGS:
        state: matrix representing the
            current observation of the environment
        weight: matrix of random weight
    Return: the action and the gradient (in this order)
    """
    probs = policy(state, weight)
    action = np.random.choice(2, p=probs[0])
    dsoftmax = softmax_grad(probs)[action, :]
    dlog = dsoftmax / probs[0, action]
    grad = state.T.dot(dlog[None, :])
    return action, grad
