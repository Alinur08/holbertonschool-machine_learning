#!/usr/bin/env python3
"""
Defines the RNNEncoder class, a GRU-based encoder layer for machine
translation.
"""
import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """
    Encodes input sequences for machine translation using a GRU-based
    recurrent neural network.
    """

    def __init__(self, vocab, embedding, units, batch):
        """
        Class constructor.
        """
        super(RNNEncoder, self).__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(
            input_dim=vocab, output_dim=embedding)
        self.gru = tf.keras.layers.GRU(
            units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )

    def initialize_hidden_state(self):
        """
        Initializes the hidden states for the RNN cell to a tensor of
        zeros.
        """
        return tf.zeros((self.batch, self.units))

    def call(self, x, initial):
        """
        Runs forward propagation for the encoder.
        """
        x = self.embedding(x)
        outputs, hidden = self.gru(x, initial_state=initial)
        return outputs, hidden
