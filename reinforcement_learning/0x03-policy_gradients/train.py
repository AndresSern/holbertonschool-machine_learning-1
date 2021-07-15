#!/usr/bin/env python3
"""REINFORCE Policy Gradients"""
import numpy as np
policy_gradient = __import__('policy_gradient').policy_gradient
policy = __import__('policy_gradient').policy


def train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False):
    """
    REINFORCE Policy Gradients
    ARGS:
        -env: initial environment
        -nb_episodes: number of episodes used for training
        -alpha: the learning rate
        -gamma: the discount factor
    """

    nA = env.action_space.n
    episode_rewards = []
    weight = np.random.rand(4, 2)

    for e in range(nb_episodes):

        state = env.reset()[None, :]
        grads = []
        rewards = []
        score = 0

        while True:
            # Uncomment to see your model train in real time (slower)
            if show_result and (e % 1000) == 0:
                env.render()

            # Sample from policy and take action in environment
            action, grad = policy_gradient(state, weight)

            next_state, reward, done, _ = env.step(action)

            grads.append(grad)
            rewards.append(reward)
            score += reward

            # Dont forget to update your old state to the new state
            state = next_state[None, :]

            if done:
                break

        # Weight update
        for i in range(len(grads)):
            # Loop through everything that happend in the episode
            # and update towards the
            # log policy gradient times **FUTURE** reward
            weight += alpha * \
                grads[i] * sum([r * (gamma ** r)
                                for t, r in enumerate(rewards[i:])])

        # Append for logging and print
        episode_rewards.append(score)
        print("EP: " + str(e) + " Score: " +
              str(score) + "    ", end="\r", flush=False)
    return episode_rewards
