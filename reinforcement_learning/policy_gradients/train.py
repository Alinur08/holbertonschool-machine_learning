#!/usr/bin/env python3
"""Module that implements full training using policy gradient."""
import numpy as np

policy_gradient = __import__('policy_gradient').policy_gradient


def train(env, nb_episodes, alpha=0.000045, gamma=0.98, show_result=False):
    """
    Implement full training with the Monte-Carlo policy gradient.
    """
    weight = np.random.rand(env.observation_space.shape[0],
                            env.action_space.n)
    scores = []

    for episode in range(nb_episodes):
        state, _ = env.reset()
        state = state.reshape(1, -1)
        grads = []
        rewards = []
        score = 0

        done = False
        while not done:
            if show_result and episode % 1000 == 0:
                env.render()

            action, grad = policy_gradient(state, weight)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            grads.append(grad)
            rewards.append(reward)
            score += reward

            state = next_state.reshape(1, -1)

        for i, grad in enumerate(grads):
            future_rewards = sum(
                r * (gamma ** t) for t, r in enumerate(rewards[i:]))
            weight += alpha * grad * future_rewards

        print("Episode: {} Score: {}".format(episode, score))
        scores.append(score)

    return scores
