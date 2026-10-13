#!/usr/bin/env python3
"""RNN Decoder for machine translation"""
import tensorflow as tf
SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """Decoder for machine translation"""

    def __init__(self, vocab, embedding, units, batch):
        """
        Class constructor.
        """
        super(RNNDecoder, self).__init__()
        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab, output_dim=embedding)
        self.gru = tf.keras.layers.GRU(
            units=units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform')
        self.F = tf.keras.layers.Dense(vocab)

    def call(self, x, s_prev, hidden_states):
        """
        Forward propagation for the RNN decoder.
        """
        units = s_prev.shape[1]
        attention = SelfAttention(units)
        context, weights = attention(s_prev, hidden_states)

        x = self.embedding(x)
        context = tf.expand_dims(context, 1)
        x = tf.concat([context, x], axis=-1)

        outputs, s = self.gru(x)

        outputs = tf.reshape(outputs, (-1, outputs.shape[2]))
        y = self.F(outputs)

        return y, s
