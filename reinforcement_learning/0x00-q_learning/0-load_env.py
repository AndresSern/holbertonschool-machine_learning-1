#!/usr/bin/env python3
"""
loads the pre-made FrozenLakeEnv evnironment from OpenAI’s gym:
"""
import gym


def load_frozen_lake(desc=None, map_name=None, is_slippery=False):
    """
    ARGS:
        -desc:{None or a list of lists} containing a custom description of
            the map to load for the environment
        -map_name:{None or a string} containing the pre-made map to load

        -is_slippery:{boolean}:to determine if the ice is slippery
    Returns: the environment
    """
    if desc is None and map_name is None:
        env = gym.make('FrozenLake8x8-v0')
    else:
        env = gym.make('FrozenLake-v0', desc=desc,
                       map_name=map_name, is_slippery=is_slippery)
        # env.reset()
    return env
