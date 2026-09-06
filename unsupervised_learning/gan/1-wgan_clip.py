#!/usr/bin/env python3
"""Wasserstein GAN with clipping."""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras


class WGAN_clip(keras.Model):
    """Simple GAN model."""

    def __init__(
        self,
        generator,
        discriminator,
        latent_generator,
        real_examples,
        batch_size=200,
        disc_iter=2,
        learning_rate=0.005,
    ):
        """Initialization."""
        super().__init__()
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter
        self.learning_rate = learning_rate
        self.beta_1 = 0.5
        self.beta_2 = 0.9

        # define the generator loss and optimizer
        self.generator.loss = lambda x: (
            -tf.math.reduce_mean(x)
        )
        self.generator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2,
        )
        self.generator.compile(
            optimizer=self.generator.optimizer,
            loss=self.generator.loss,
        )

        # define the discriminator loss and optimizer
        self.discriminator.loss = lambda x, y: (
            tf.math.reduce_mean(x) - tf.math.reduce_mean(y)
        )
        self.discriminator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2,
        )
        self.discriminator.compile(
            optimizer=self.discriminator.optimizer,
            loss=self.discriminator.loss,
        )

    # generator of fake samples of size batch_size
    def get_fake_sample(self, size=None, training=False):
        """Generate fake samples."""
        if size is None:
            size = self.batch_size
        return self.generator(self.latent_generator(size), training=training)

    # generator of real samples of size batch_size
    def get_real_sample(self, size=None):
        """Generate real samples."""
        if size is None:
            size = self.batch_size
        sorted_indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices = tf.random.shuffle(sorted_indices)[:size]
        return tf.gather(self.real_examples, random_indices)

    # overloading train_step()
    def train_step(self, useless_argument):
        """Train step."""
        for _ in range(self.disc_iter):
            with tf.GradientTape() as discr_tape:
                real_sample = self.get_real_sample()
                fake_sample = self.get_fake_sample()
                real_output = self.discriminator(real_sample, training=True)
                fake_output = self.discriminator(fake_sample, training=True)
                discr_loss = self.discriminator.loss(fake_output, real_output)
            discr_gradient = discr_tape.gradient(
                discr_loss, self.discriminator.trainable_variables
            )
            self.discriminator.optimizer.apply_gradients(
                zip(discr_gradient, self.discriminator.trainable_variables)
            )
            # clip discriminator weights to [-1, 1]
            for var in self.discriminator.trainable_variables:
                var.assign(tf.clip_by_value(var, -1.0, 1.0))

        with tf.GradientTape() as gen_tape:
            fake_sample = self.get_fake_sample(training=True)
            gen_output = self.discriminator(fake_sample, training=False)
            gen_loss = self.generator.loss(gen_output)
        gen_gradient = gen_tape.gradient(
            gen_loss, self.generator.trainable_variables
        )
        self.generator.optimizer.apply_gradients(
            zip(gen_gradient, self.generator.trainable_variables)
        )

        return {"discr_loss": discr_loss, "gen_loss": gen_loss}
