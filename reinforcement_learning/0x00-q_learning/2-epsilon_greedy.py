#!/usr/bin/env python3
"""
uses epsilon-greedy to determine the next action
"""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """
    ARGS:
        -Q:{numpy.ndarray}: containing the q-table
        -state: is the current state
        -epsilon is the epsilon to use for the calculation

    Returns: the next action index
    """
    # we sample a float from a uniform distribution over 0 and 1
    # if the sampled flaot is less than the exploration proba
    #     the agent selects arandom action
    # else
    #      he exploits his knowledge using the bellman equation
    if np.random.uniform(0, 1) < epsilon:
        A = np.random.randint(0, Q.shape[1])

    else:
        A = np.argmax(Q[state])

    return A
