#!/usr/bin/env python3
"""
initializes the Q-table:
"""
import numpy as np


def q_init(env):
    '''
    ARGS:
    env is the FrozenLakeEnv instance
    Returns: the Q-table as a numpy.ndarray of zeros
    '''

    # a = env.desc.shape[0]
    # b = env.P[0][0]
    # a = a *a
    # b = len(b[0])
    # qtable = np.zeros((a,b))

    action_size = env.action_space.n
    # print(‘Action Space: ‘, action_size)
    state_size = env.observation_space.n
    qtable = np.zeros((state_size, action_size))
    return qtable
