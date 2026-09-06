#!/usr/bin/env python3
"""Gensim to keras"""
import tensorflow as tf


def gensim_to_keras(model):
    """
    Converts a gensim word2vec model to a trainable keras Embedding layer.
    """
    weights = model.wv.vectors
    vocab_size, vector_size = weights.shape

    return tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=vector_size,
        weights=[weights],
        trainable=True,
    )
