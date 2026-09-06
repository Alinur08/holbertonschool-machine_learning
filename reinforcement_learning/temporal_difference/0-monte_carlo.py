#!/usr/bin/env python3
"""
Monte Carlo Algorithm
"""
import numpy as np


def monte_carlo(env, V, policy, episodes=5000, max_steps=100, alpha=0.1,
                gamma=0.99):
    """
    Monte Carlo algorithm to estimate the value function
    """
    for episode in range(episodes):
        state = env.reset()[0]
        episode_data = []

        for step in range(max_steps):
            action = policy(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            episode_data.append((state, reward))

            if terminated or truncated:
                break

            state = next_state

        G = 0
        episode_data = np.array(episode_data, dtype=int)

        for state, reward in reversed(episode_data):
            G = reward + gamma * G

            if state not in episode_data[:episode, 0]:
                V[state] += alpha * (G - V[state])

    return V
