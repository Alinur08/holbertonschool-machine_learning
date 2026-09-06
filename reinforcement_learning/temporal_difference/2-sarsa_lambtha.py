#!/usr/bin/env python3
"""Module that implements the SARSA(λ) algorithm for Q-value estimation."""
import numpy as np


def epsilon_greedy(Q, state, epsilon):
    """Choose an action using the epsilon-greedy policy.

    Args:
        Q (numpy.ndarray): shape (s, a), the Q table
        state (int): the current state
        epsilon (float): the epsilon value to use

    Returns:
        int: the chosen action
    """
    p = np.random.uniform(0, 1)
    if p < epsilon:
        return np.random.randint(Q.shape[1])
    return np.argmax(Q[state])


def sarsa_lambtha(env, Q, lambtha, episodes=5000, max_steps=100,
                  alpha=0.1, gamma=0.99, epsilon=1, min_epsilon=0.1,
                  epsilon_decay=0.05):
    """Perform the SARSA(λ) algorithm to estimate a Q table.

    Args:
        env: the environment instance
        Q (numpy.ndarray): shape (s, a), the Q table
        lambtha (float): the eligibility trace factor
        episodes (int): total number of episodes to train over
        max_steps (int): maximum number of steps per episode
        alpha (float): the learning rate
        gamma (float): the discount rate
        epsilon (float): the initial threshold for epsilon greedy
        min_epsilon (float): the minimum value epsilon should decay to
        epsilon_decay (float): the decay rate for updating epsilon
            between episodes

    Returns:
        numpy.ndarray: the updated Q table
    """
    initial_epsilon = epsilon

    for ep in range(episodes):
        state, _ = env.reset()
        action = epsilon_greedy(Q, state, epsilon)
        eligibility = np.zeros(Q.shape)

        for step in range(max_steps):
            next_state, reward, done, truncated, _ = env.step(action)
            next_action = epsilon_greedy(Q, next_state, epsilon)

            td_error = (reward + gamma * Q[next_state, next_action]
                        - Q[state, action])
            eligibility[state, action] += 1

            Q += alpha * td_error * eligibility
            eligibility *= gamma * lambtha

            if done or truncated:
                break
            state = next_state
            action = next_action

        epsilon = (min_epsilon + (initial_epsilon - min_epsilon)
                   * np.exp(-epsilon_decay * ep))

    return Q
