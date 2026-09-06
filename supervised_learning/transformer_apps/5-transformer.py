#!/usr/bin/env python3
"""Transformer Network for machine translation"""
import numpy as np
import tensorflow as tf


def positional_encoding(max_seq_len, dm):
    """Calculates the positional encoding for a transformer

    Args:
        max_seq_len: integer representing the maximum sequence length
        dm: the model depth

    Returns:
        a numpy.ndarray of shape (max_seq_len, dm) containing the
        positional encoding vectors
    """
    PE = np.zeros((max_seq_len, dm))
    positions = np.arange(max_seq_len)[:, np.newaxis]
    div_term = np.power(
        10000, (2 * (np.arange(dm) // 2)) / np.float32(dm))

    angles = positions / div_term

    PE[:, 0::2] = np.sin(angles[:, 0::2])
    PE[:, 1::2] = np.cos(angles[:, 1::2])

    # Cast to float32 so it can be added to float32 embedding tensors
    return PE.astype(np.float32)


def sdp_attention(Q, K, V, mask=None):
    """Calculates the scaled dot product attention"""
    matmul_qk = tf.matmul(Q, K, transpose_b=True)

    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    weights = tf.nn.softmax(scaled_attention_logits, axis=-1)

    output = tf.matmul(weights, V)

    return output, weights


class MultiHeadAttention(tf.keras.layers.Layer):
    """Performs multi head attention"""

    def __init__(self, dm, h):
        """Class constructor"""
        super(MultiHeadAttention, self).__init__()
        self.h = h
        self.dm = dm
        self.depth = dm // h

        self.Wq = tf.keras.layers.Dense(dm)
        self.Wk = tf.keras.layers.Dense(dm)
        self.Wv = tf.keras.layers.Dense(dm)
        self.linear = tf.keras.layers.Dense(dm)

    def split_heads(self, x, batch_size):
        """Splits the last dimension into (h, depth) and transposes"""
        x = tf.reshape(x, (batch_size, -1, self.h, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, Q, K, V, mask):
        """Performs multi head attention"""
        batch_size = tf.shape(Q)[0]

        Q = self.Wq(Q)
        K = self.Wk(K)
        V = self.Wv(V)

        Q = self.split_heads(Q, batch_size)
        K = self.split_heads(K, batch_size)
        V = self.split_heads(V, batch_size)

        scaled_attention, weights = sdp_attention(Q, K, V, mask)

        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(
            scaled_attention, (batch_size, -1, self.dm))

        output = self.linear(concat_attention)

        return output, weights


class EncoderBlock(tf.keras.layers.Layer):
    """Creates an encoder block for a transformer"""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """Class constructor"""
        super(EncoderBlock, self).__init__()
        self.mha = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(
            hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask=None):
        """Creates an encoder block output"""
        attn_output, _ = self.mha(x, x, x, mask)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(x + attn_output)

        hidden_output = self.dense_hidden(out1)
        dense_output = self.dense_output(hidden_output)
        dense_output = self.dropout2(dense_output, training=training)
        out2 = self.layernorm2(out1 + dense_output)

        return out2


class DecoderBlock(tf.keras.layers.Layer):
    """Creates a decoder block for a transformer"""

    def __init__(self, dm, h, hidden, drop_rate=0.1):
        """Class constructor"""
        super(DecoderBlock, self).__init__()
        self.mha1 = MultiHeadAttention(dm, h)
        self.mha2 = MultiHeadAttention(dm, h)
        self.dense_hidden = tf.keras.layers.Dense(
            hidden, activation='relu')
        self.dense_output = tf.keras.layers.Dense(dm)
        self.layernorm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.layernorm3 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = tf.keras.layers.Dropout(drop_rate)
        self.dropout2 = tf.keras.layers.Dropout(drop_rate)
        self.dropout3 = tf.keras.layers.Dropout(drop_rate)

    def call(
            self, x, encoder_output, training, look_ahead_mask,
            padding_mask):
        """Creates a decoder block output"""
        attn1, _ = self.mha1(x, x, x, look_ahead_mask)
        attn1 = self.dropout1(attn1, training=training)
        out1 = self.layernorm1(x + attn1)

        attn2, _ = self.mha2(
            out1, encoder_output, encoder_output, padding_mask)
        attn2 = self.dropout2(attn2, training=training)
        out2 = self.layernorm2(out1 + attn2)

        hidden_output = self.dense_hidden(out2)
        dense_output = self.dense_output(hidden_output)
        dense_output = self.dropout3(dense_output, training=training)
        out3 = self.layernorm3(out2 + dense_output)

        return out3


class Encoder(tf.keras.layers.Layer):
    """Creates the encoder for a transformer"""

    def __init__(self, N, dm, h, hidden, input_vocab, max_seq_len,
                 drop_rate=0.1):
        """Class constructor"""
        super(Encoder, self).__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(
            input_dim=input_vocab, output_dim=dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [
            EncoderBlock(dm, h, hidden, drop_rate) for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, training, mask):
        """Creates the encoder output"""
        seq_len = tf.shape(x)[1]

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[:seq_len]

        x = self.dropout(x, training=training)

        for block in self.blocks:
            x = block(x, training, mask)

        return x


class Decoder(tf.keras.layers.Layer):
    """Creates the decoder for a transformer"""

    def __init__(self, N, dm, h, hidden, target_vocab, max_seq_len,
                 drop_rate=0.1):
        """Class constructor"""
        super(Decoder, self).__init__()
        self.N = N
        self.dm = dm
        self.embedding = tf.keras.layers.Embedding(
            input_dim=target_vocab, output_dim=dm)
        self.positional_encoding = positional_encoding(max_seq_len, dm)
        self.blocks = [
            DecoderBlock(dm, h, hidden, drop_rate) for _ in range(N)]
        self.dropout = tf.keras.layers.Dropout(drop_rate)

    def call(self, x, encoder_output, training, look_ahead_mask,
              padding_mask):
        """Creates the decoder output"""
        seq_len = tf.shape(x)[1]

        x = self.embedding(x)
        x *= tf.math.sqrt(tf.cast(self.dm, tf.float32))
        x += self.positional_encoding[:seq_len]

        x = self.dropout(x, training=training)

        for block in self.blocks:
            x = block(
                x, encoder_output, training, look_ahead_mask,
                padding_mask)

        return x


class Transformer(tf.keras.Model):
    """Creates a transformer network"""

    def __init__(
            self, N, dm, h, hidden, input_vocab, target_vocab,
            max_seq_input, max_seq_target, drop_rate=0.1):
        """Class constructor"""
        super(Transformer, self).__init__()
        self.encoder = Encoder(
            N, dm, h, hidden, input_vocab, max_seq_input, drop_rate)
        self.decoder = Decoder(
            N, dm, h, hidden, target_vocab, max_seq_target, drop_rate)
        self.linear = tf.keras.layers.Dense(target_vocab)

    def call(
            self, inputs, target, training, encoder_mask,
            look_ahead_mask, decoder_mask):
        """Creates the transformer output"""
        encoder_output = self.encoder(inputs, training, encoder_mask)

        decoder_output = self.decoder(
            target, encoder_output, training, look_ahead_mask,
            decoder_mask)

        output = self.linear(decoder_output)

        return output
