#!/usr/bin/env python3
"""Convolutional autoencoder"""
import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """Creates a convolutional autoencoder"""

    # Encoder
    encoder_inputs = keras.Input(shape=input_dims)
    x = encoder_inputs
    for f in filters:
        x = keras.layers.Conv2D(f, (3, 3), padding='same',
                                activation='relu')(x)
        x = keras.layers.MaxPooling2D((2, 2), padding='same')(x)
    encoder = keras.Model(encoder_inputs, x)

    # Decoder
    decoder_inputs = keras.Input(shape=latent_dims)
    x = decoder_inputs
    rev_filters = list(reversed(filters))
    for f in rev_filters[:-1]:
        x = keras.layers.Conv2D(f, (3, 3), padding='same',
                                activation='relu')(x)
        x = keras.layers.UpSampling2D((2, 2))(x)
    # second to last convolution: valid padding
    x = keras.layers.Conv2D(rev_filters[-1], (3, 3), padding='valid',
                            activation='relu')(x)
    x = keras.layers.UpSampling2D((2, 2))(x)
    # last convolution: same number of channels as input, sigmoid,
    # no upsampling
    decoder_outputs = keras.layers.Conv2D(
        input_dims[-1], (3, 3), padding='same', activation='sigmoid'
    )(x)
    decoder = keras.Model(decoder_inputs, decoder_outputs)

    # Full autoencoder
    auto_inputs = keras.Input(shape=input_dims)
    encoded = encoder(auto_inputs)
    decoded = decoder(encoded)
    auto = keras.Model(auto_inputs, decoded)

    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
