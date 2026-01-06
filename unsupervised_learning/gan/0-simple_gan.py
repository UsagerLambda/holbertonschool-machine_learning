#!/usr/bin/env python3

"""Simple GAN implementation module."""

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras


class Simple_GAN(keras.Model):
    """Simple Generative Adversarial Network (GAN) implementation.

    This class implements a basic GAN with a generator and discriminator
    trained adversarially. The generator creates fake samples to fool
    the discriminator, while the discriminator learns to distinguish
    real from fake samples.
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
        """Initialize the Simple GAN.

        Args:
            generator: Neural network that generates fake samples
            discriminator: Neural network that classifies real vs fake
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

        # define the generator loss and optimizer:
        self.generator.loss = lambda x: tf.keras.losses.MeanSquaredError()(
            x, tf.ones(x.shape)
        )
        self.generator.optimizer = keras.optimizers.Adam(
            learning_rate=self.learning_rate,
            beta_1=self.beta_1,
            beta_2=self.beta_2,
        )
        self.generator.compile(
            optimizer=generator.optimizer, loss=generator.loss
        )

        # define the discriminator loss and optimizer:
        self.discriminator.loss = (
            lambda x, y: tf.keras.losses.MeanSquaredError()(
                x, tf.ones(x.shape)
            )
            + tf.keras.losses.MeanSquaredError()(y, -1 * tf.ones(y.shape))
        )
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
            # with GradientTape() as g,
            # enregistre les opérations pour calculer les gradients
            with tf.GradientTape() as g:
                # Prédiction du discriminateur sur de vraies données (~1)
                real = self.discriminator(self.get_real_sample())
                # Prédiction du discriminateur sur de fausses données (~-1)
                fake = self.discriminator(self.get_fake_sample())
                # Calcule la loss du discriminateur
                # Plus la loss est proche de 0,
                # mieux le discriminateur distingue vrai/faux
                discr_loss = self.discriminator.loss(real, fake)
                # Calcule les gradients de la loss par rapport aux poids
                # du discriminateur uniquement
                gradient = g.gradient(
                    discr_loss, self.discriminator.trainable_variables
                )
            # Applique le gradients pour mettre à jour les poids
            self.discriminator.optimizer.apply_gradients(
                zip(gradient, self.discriminator.trainable_variables)
            )

        # enregistre les opérations pour calculer les gradients
        with tf.GradientTape() as g:
            # training=True pour que GradientTape enregistre aussi les
            # opérations du générateur (pas juste le discriminateur)
            # ça permet aux gradients de remonter
            # loss -> discriminateur -> générateur
            # Sans le training=True, le générateur est traité comme une
            # "boîte noire" qui produit des images
            fake = self.discriminator(self.get_fake_sample(), training=True)
            # Loss du générateur, il veut que la prédiction
            # du discriminateur soit le plus proche de 1
            # (discriminateur pense que l'image générer est vraie)
            gen_loss = self.generator.loss(fake)
            # Calcule les gradients de la loss par rapport aux poids
            # du générateur
            gradient = g.gradient(gen_loss, self.generator.trainable_variables)
        # Applique les gradients pour mettre à jour les poids du générateur
        self.generator.optimizer.apply_gradients(
            zip(gradient, self.generator.trainable_variables)
        )

        return {"discr_loss": discr_loss, "gen_loss": gen_loss}
