#!/usr/bin/env python3

"""Wasserstein GAN with weight clipping implementation module."""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras


class WGAN_clip(keras.Model):
    """Wasserstein GAN (WGAN) with weight clipping implementation.

    This class implements a WGAN which uses the Wasserstein distance
    as the loss function and applies weight clipping to enforce the
    Lipschitz constraint on the discriminator (critic).
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
    ):
        """Initialize the WGAN with weight clipping.

        Args:
            generator: Neural network that generates fake samples
            discriminator: Neural network (critic) that scores samples
            latent_generator: Function that generates random latent vectors
            real_examples: Dataset of real samples
            batch_size: Number of samples per training batch (default: 200)
            disc_iter: Number of discriminator iterations per generator
                iteration (default: 2)
            learning_rate: Learning rate for Adam optimizer (default: 0.005)
        """
        super().__init__()  # run the __init__ of keras.Model first.
        self.latent_generator = latent_generator
        self.real_examples = real_examples
        self.generator = generator
        self.discriminator = discriminator
        self.batch_size = batch_size
        self.disc_iter = disc_iter

        self.learning_rate = learning_rate
        self.beta_1 = 0.5  # standard value, but can be changed if necessary
        self.beta_2 = 0.9  # standard value, but can be changed if necessary

        # reçois les scores du discriminateur sur les fausses images
        # fait la moyenne et inverse le signe
        # Score haut = discriminateur pense que c'est vrai = bien pour G
        # Score bas = discriminateur a détecté la fausse = pas bien pour G
        # Le moins transforme "maximiser le score" en "minimiser la loss"
        self.generator.loss = lambda x: -tf.math.reduce_mean(x)
        self.generator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2,
        )
        self.generator.compile(
            optimizer=generator.optimizer, loss=generator.loss
        )

        # fake - real = loss
        # Plus la loss est négative plus le discriminateur est bon
        # à l'inverse s'il est positif il fais de mauvaise prédiction
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
                real = self.discriminator(self.get_real_sample())
                fake = self.discriminator(self.get_fake_sample())
                discr_loss = self.discriminator.loss(real, fake)
                gradient = g.gradient(
                    discr_loss, self.discriminator.trainable_variables
                )
            self.discriminator.optimizer.apply_gradients(
                zip(gradient, self.discriminator.trainable_variables)
            )

            # Limite les poids entre -1 et 1
            for layer in self.discriminator.layers:
                for i, w in enumerate(layer.weights):
                    layer.weights[i].assign(tf.clip_by_value(w, -1, 1))

        with tf.GradientTape() as g:
            fake = self.discriminator(self.get_fake_sample(), training=True)
            gen_loss = self.generator.loss(fake)
            gradient = g.gradient(gen_loss, self.generator.trainable_variables)
        self.generator.optimizer.apply_gradients(
            zip(gradient, self.generator.trainable_variables)
        )

        return {"discr_loss": discr_loss, "gen_loss": gen_loss}
