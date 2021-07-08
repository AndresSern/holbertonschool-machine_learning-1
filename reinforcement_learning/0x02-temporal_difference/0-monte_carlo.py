#!/usr/bin/env python3

import numpy as np
import gym


def generate_episode(env, policy, max_steps):
    """test"""
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
    """test"""

    nA = env.action_space.n
    N = {new_list: [] for new_list in range(1, nA+1)}
    for i in range(1, episodes+1):

        # Generate episopde
        episode = generate_episode(env, policy, max_steps)

        # Update Q values
        states, rewards = zip(*episode)
        discounts = np.array([gamma ** i for i in range(len(rewards) + 1)])

        for idx in range(len(episode[0])):

            G = sum(rewards[idx:] * discounts[:-(idx+1)])

            V[episode[idx][0]] = V[episode[idx][0]] + \
                alpha * (G - V[episode[idx][0]])

    return V
