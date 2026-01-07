#!/usr/bin/env python3

"""Wasserstein GAN with Gradient Penalty implementation module."""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras


class WGAN_GP(keras.Model):
    """Wasserstein GAN with Gradient Penalty (WGAN-GP) implementation.

    This class implements a WGAN-GP which uses the Wasserstein distance
    as the loss function and applies gradient penalty instead of weight
    clipping to enforce the Lipschitz constraint on the discriminator.
    """

    def __init__(
        self,
        generator,
        discriminator,
        latent_generator,
        real_examples,
        batch_size=200,
        disc_iter=2,
        learning_rate=0.005,
        lambda_gp=10,
    ):
        """Initialize the WGAN with gradient penalty.

        Args:
            generator: Neural network that generates fake samples
            discriminator: Neural network (critic) that scores samples
            latent_generator: Function that generates random latent vectors
            real_examples: Dataset of real samples
            batch_size: Number of samples per training batch (default: 200)
            disc_iter: Number of discriminator iterations per generator
                iteration (default: 2)
            learning_rate: Learning rate for Adam optimizer (default: 0.005)
            lambda_gp: Weight for the gradient penalty term (default: 10)
        """
        super().__init__()  # run the __init__ of keras.Model first.
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter

        self.learning_rate = learning_rate
        self.beta_1 = 0.3  # standard value, but can be changed if necessary
        self.beta_2 = 0.9  # standard value, but can be changed if necessary

        self.lambda_gp = lambda_gp
        self.dims = self.real_examples.shape
        self.len_dims = tf.size(self.dims)
        self.axis = tf.range(1, self.len_dims, delta=1, dtype="int32")
        self.scal_shape = self.dims.as_list()
        self.scal_shape[0] = self.batch_size
        for i in range(1, self.len_dims):
            self.scal_shape[i] = 1
        self.scal_shape = tf.convert_to_tensor(self.scal_shape)

        # define the generator loss and optimizer:
        self.generator.loss = lambda x: -tf.math.reduce_mean(x)
        self.generator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2,
        )
        self.generator.compile(
            optimizer=generator.optimizer, loss=generator.loss
        )

        # define the discriminator loss and optimizer:
        self.discriminator.loss = lambda x, y: tf.math.reduce_mean(
            y
        ) - tf.math.reduce_mean(x)
        self.discriminator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2,
        )
        self.discriminator.compile(
            optimizer=discriminator.optimizer, loss=discriminator.loss
        )

    def get_fake_sample(self, size=None, training=False):
        """Generate fake samples using the generator.

        Args:
            size: Number of samples to generate (default: batch_size)
            training: Whether to run generator in training mode

        Returns:
            Tensor of fake samples generated from random latent vectors
        """
        if not size:
            size = self.batch_size
        return self.generator(self.latent_generator(size), training=training)

    def get_real_sample(self, size=None):
        """Get real samples from the dataset.

        Args:
            size: Number of samples to retrieve (default: batch_size)

        Returns:
            Tensor of real samples randomly selected from the dataset
        """
        if not size:
            size = self.batch_size
        sorted_indices = tf.range(tf.shape(self.real_examples)[0])
        random_indices = tf.random.shuffle(sorted_indices)[:size]
        return tf.gather(self.real_examples, random_indices)

    def get_interpolated_sample(self, real_sample, fake_sample):
        """Generate interpolated samples between real and fake samples.

        Creates random linear interpolations between real and fake samples
        for computing the gradient penalty.

        Args:
            real_sample: Tensor of real samples
            fake_sample: Tensor of fake samples

        Returns:
            Tensor of interpolated samples (u * real + (1-u) * fake)
        """
        u = tf.random.uniform(self.scal_shape)
        v = tf.ones(self.scal_shape) - u
        return u * real_sample + v * fake_sample

    def gradient_penalty(self, interpolated_sample):
        """Compute the gradient penalty for WGAN-GP.

        Calculates the gradient penalty to enforce the Lipschitz constraint
        on the discriminator. The penalty is based on the norm of gradients
        with respect to interpolated samples.

        Args:
            interpolated_sample: Tensor of interpolated samples between
                real and fake samples

        Returns:
            Scalar tensor representing the gradient penalty loss
        """
        with tf.GradientTape() as gp_tape:
            gp_tape.watch(interpolated_sample)
            pred = self.discriminator(interpolated_sample, training=True)
        grads = gp_tape.gradient(pred, [interpolated_sample])[0]
        norm = tf.sqrt(tf.reduce_sum(tf.square(grads), axis=self.axis))
        return tf.reduce_mean((norm - 1.0) ** 2)

    def train_step(self, useless_argument):
        """Execute one training step of the GAN.

        Trains the discriminator disc_iter times, then trains the generator
        once. This implements the adversarial training process where the
        discriminator learns to distinguish real from fake samples, and the
        generator learns to create samples that fool the discriminator.

        Args:
            useless_argument: Unused argument (required by Keras interface)

        Returns:
            Dictionary containing 'discr_loss' and 'gen_loss' metrics
        """
        # Entraine le discriminateur disc_iter fois
        for _ in range(self.disc_iter):
            with tf.GradientTape() as g:
                real_s = self.get_real_sample()
                fake_s = self.get_fake_sample()
                real = self.discriminator(real_s)
                fake = self.discriminator(fake_s)
                temp_loss = self.discriminator.loss(real, fake)
                interpolated = self.get_interpolated_sample(real_s, fake_s)
                gp = self.gradient_penalty(interpolated)
                discr_loss = temp_loss + self.lambda_gp * gp
            gradient = g.gradient(
                discr_loss, self.discriminator.trainable_variables
            )
            self.discriminator.optimizer.apply_gradients(
                zip(gradient, self.discriminator.trainable_variables)
            )

        with tf.GradientTape() as g:
            fake = self.discriminator(self.get_fake_sample(), training=True)
            gen_loss = self.generator.loss(fake)
            gradient = g.gradient(gen_loss, self.generator.trainable_variables)
        self.generator.optimizer.apply_gradients(
            zip(gradient, self.generator.trainable_variables)
        )

        return {"discr_loss": discr_loss, "gen_loss": gen_loss}
