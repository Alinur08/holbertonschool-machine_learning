#!/usr/bin/env python3
"""Create masks module"""
import tensorflow as tf


def create_masks(inputs, target):
    """
    Creates all masks needed for training/validation.
    """
    # Padding mask for the encoder (masks padding tokens in the input)
    encoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    encoder_mask = encoder_mask[:, tf.newaxis, tf.newaxis, :]

    # Padding mask for the 2nd attention block in the decoder
    # (based on the encoder input, same as encoder_mask)
    decoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    decoder_mask = decoder_mask[:, tf.newaxis, tf.newaxis, :]

    # Look ahead mask to prevent the decoder from seeing future tokens
    seq_len_out = tf.shape(target)[1]
    look_ahead_mask = 1 - tf.linalg.band_part(
        tf.ones((seq_len_out, seq_len_out)), -1, 0)

    # Padding mask for the target (used in the 1st attention block)
    target_padding_mask = tf.cast(tf.math.equal(target, 0), tf.float32)
    target_padding_mask = target_padding_mask[:, tf.newaxis, tf.newaxis, :]

    # Combined mask takes the max of the look ahead and target padding masks
    combined_mask = tf.maximum(target_padding_mask, look_ahead_mask)

    return encoder_mask, combined_mask, decoder_mask
