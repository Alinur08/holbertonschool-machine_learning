#!/usr/bin/env python3
"""Module that implements the TD(λ) algorithm for value estimation."""
import numpy as np


def td_lambtha(env, V, policy, lambtha, episodes=5000, max_steps=100,
               alpha=0.1, gamma=0.99):
    """Perform the TD(λ) algorithm to estimate a value function.

    Args:
        env: the environment instance
        V (numpy.ndarray): shape (s,), the value estimate
        policy: function that takes a state and returns the next action
        lambtha (float): the eligibility trace factor
        episodes (int): total number of episodes to train over
        max_steps (int): maximum number of steps per episode
        alpha (float): the learning rate
        gamma (float): the discount rate

    Returns:
        numpy.ndarray: the updated value estimate V
    """
    for ep in range(episodes):
        state, _ = env.reset()
        eligibility = np.zeros(V.shape[0])

        for step in range(max_steps):
            action = policy(state)
            next_state, reward, done, truncated, _ = env.step(action)

            td_error = reward + gamma * V[next_state] - V[state]
            eligibility[state] += 1

            V += alpha * td_error * eligibility
            eligibility *= gamma * lambtha

            if done or truncated:
                break
            state = next_state

    return V
