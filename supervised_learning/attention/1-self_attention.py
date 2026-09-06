#!/usr/bin/env python3
"""Self Attention for machine translation"""
import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """Calculates the attention for machine translation"""

    def __init__(self, units):
        """
        Class constructor.
        """
        super(SelfAttention, self).__init__()
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """
        Forward propagation for the self attention layer.
        """
        s_prev_expanded = tf.expand_dims(s_prev, 1)
        score_input = self.W(s_prev_expanded) + self.U(hidden_states)
        score = self.V(tf.nn.tanh(score_input))

        weights = tf.nn.softmax(score, axis=1)

        context = weights * hidden_states
        context = tf.reduce_sum(context, axis=1)

        return context, weights
