#!/usr/bin/env python3
"""
On-policy First Visit Monte Carlo Control with the incremental mean
"""
import numpy as np
import gym


def generate_episode(env, policy, max_steps):
    """
    generate episode
    Args:
        -env is the openAI environment instance

        -policy is a function that takes in a state and
            returns the next action to take
        -max_steps is the maximum number of steps per episode
    Returns : list of tuple  state and reward
    """
    state = env.reset()
    episode = []
    for i in range(max_steps):
        action = policy(state)
        next_state, reward, done, info = env.step(action)
        episode.append((state, reward))
        state = next_state
        if done:
            break
    return episode


def monte_carlo(env,
                V,
                policy,
                episodes=5000,
                max_steps=100,
                alpha=0.1,
                gamma=0.99):
    """
    Args:
        -env is the openAI environment instance
        -V is a numpy.ndarray of shape (s,)
            containing the value estimate
        -policy is a function that takes in a state and
            returns the next action to take
        -episodes is the total number of episodes to train over
        -max_steps is the maximum number of steps per episode
        -alpha is the learning rate
        -gamma is the discount rate

    Returns: V, the updated value estimate
    """
    returns = set()
    for i in range(1, episodes+1):

        # Generate episopde
        episode = generate_episode(env, policy, max_steps)

        # Update Q values
        states, rewards = zip(*episode)
        discounts = np.array([gamma ** i for i in range(len(rewards) + 1)])

        for idx in range(len(episode[0])-1, -1, -1):

            G = sum(rewards[idx:] * discounts[:-(idx+1)])
            if not episode[idx][0] in returns:
                V[episode[idx][0]] = V[episode[idx][0]] + \
                    alpha * (G - V[episode[idx][0]])
            returns.add(episode[idx][0])
    return V
