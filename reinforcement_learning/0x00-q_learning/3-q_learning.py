#!/usr/bin/env python3
import numpy as np
def train(env, Q, episodes=5000, max_steps=100, alpha=0.1,
          gamma=0.99, epsilon=1, min_epsilon=0.1, epsilon_decay=0.05):
    rewards_per_episode =[]
    init_epsilon = epsilon
    for e in range(episodes):
        #we initialize the first state of the episode
        current_state = env.reset()
        done = False
        
        #sum the rewards that the agent gets from the environment
        total_episode_reward = 0
        
        for i in range(max_steps): 
            # we sample a float from a uniform distribution over 0 and 1
            # if the sampled flaot is less than the exploration proba
            #     the agent selects arandom action
            # else
            #     he exploits his knowledge using the bellman equation 
            
            if np.random.uniform(0,1) < epsilon:
                action = np.random.randint(0, Q.shape[1])

            else:
                action = np.argmax(Q[current_state])
            
            # The environment runs the chosen action and returns
            # the next state, a reward and true if the epiosed is ended.
            next_state, reward, done, _ = env.step(action)

            if done and reward == 0:
                reward = -1
            
            # We update our Q-table
            Q[current_state, action] = (1-epsilon) * Q[current_state, action] +alpha*(reward + gamma*max(Q[next_state]))
            total_episode_reward = total_episode_reward + reward
            # If the episode is finished, we leave the for loop
            if done:
                break
            current_state = next_state
        #We update the epsilon
        epsilon = min_epsilon + (init_epsilon - min_epsilon) *\
            np.exp(-epsilon_decay * e)
        rewards_per_episode.append(total_episode_reward)
    return Q,rewards_per_episode