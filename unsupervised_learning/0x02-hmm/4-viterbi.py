#!/usr/bin/env python3
"""calculates the most likely sequence of hidden states
for a hidden markov model:"""
import numpy as np


def viterbi(Observation, Emission, Transition, Initial):
    """
    ARGS:
    *Observation is a numpy.ndarray of shape (T,)
    that contains the index of the observation
        -T is the number of observations

    *Emission is a numpy.ndarray of shape (N, M) containing
    the emission probability of a specific observation given a hidden state
        -Emission[i, j] is the probability of observing j
            given the hidden state i
        -N is the number of hidden states
        -M is the number of all possible observations

    *Transition is a 2D numpy.ndarray of shape (N, N)
    containing the transition probabilities
        -Transition[i, j] is the probability of
            transitioning from the hidden state i to j
        -Initial a numpy.ndarray of shape (N, 1)
            containing the probability of starting in a particular hidden state

    Returns: path, P, or None, None on failure
        -path is the a list of length T containing
            the most likely sequence of hidden states
        -P is the probability of obtaining the path sequence
    """
    T = Observation.shape[0]
    N = Emission.shape[0]
    F = np.zeros((N, T))
    all_prev_states = np.zeros((N, T))

    ob_ind = Observation[0]
    # intial step of the Viterbi algorithm.
    F[:, 0] = np.multiply(Initial.T, Emission[:, ob_ind])
    all_prev_states[:, 0] = 0
    for i in range(1, T):
        for j in range(N):
            # Runs all others steps of the Viterbi algorithm.
            ob_ind = Observation[i]
            X = Transition[:, j] * F[:, i - 1]
            alpha = Emission[j, ob_ind]
            F[j, i] = np.max(alpha * X)
            all_prev_states[j, i] = np.argmax(alpha * X)
    # Traces backwards to get the maximum likelihood sequence.
    P = F[:, T - 1].max()

    state = int(np.argmax(F[:, T - 1]))
    state_sequence = []
    state_sequence.append(state)

    for i in range(T - 1, 0, -1):
        state_sequence.append(int(all_prev_states[state, i]))
        state = int(all_prev_states[state, i])
    state_sequence.reverse()

    return state_sequence, P
