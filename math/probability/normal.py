#!/usr/bin/env python3
"""Module pour la distribution Normal."""

e = 2.7182818285
pi = 3.1415926536


class Normal:
    """Classe représentant une distribution Normal."""

    def __init__(self, data=None, mean=0., stddev=1.):
        """
        Initialise une instance de la distribution normale.

        Args:
            data (list, optionnel): Liste de données pour estimer
            la moyenne et l'écart-type.
                Si fourni, 'mean' et 'stddev' sont ignorés.
            mean (float, optionnel): Moyenne de la distribution. Par défaut à 0
            stddev (float, optionnel): Écart-type de la distribution.
                Par défaut à 1.

        Raises:
            TypeError: Si 'data' n'est pas une liste.
            ValueError: Si 'data' contient moins de deux
                valeurs ou si 'stddev' est négatif.
        """
        self.mean = float(mean)
        self.stddev = float(stddev)

        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")

        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")

            if len(data) < 2:
                raise ValueError(
                    "data must contain multiple values")

            self.mean = float(sum(data) / len(data))

            first = 0
            for x in data:
                first += (x - self.mean) ** 2
            variance = (1 / len(data)) * first

            self.stddev = float(variance ** 0.5)

    def z_score(self, x):
        """
        Calculate the z-score of a value x.

        Args:
            x (float): Valeur à transformer en score z.

        Returns:
            float: Score z correspondant à x.
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """
        Calculate the x value from a z-score.

        Args:
            z (float): Score z à transformer en valeur x.

        Returns:
            float: Valeur x correspondant au score z.
        """
        return z * self.stddev + self.mean

    def pdf(self, x):
        """
        Calculate the probability density function (PDF) for a value x.

        Calcule la densité de probabilité (PDF) pour une valeur x
        selon la distribution normale.

        Args:
            x (float): Valeur pour laquelle calculer la densité.

        Returns:
            float: Densité de probabilité de x.
        """
        first = 1 / (self.stddev * ((2 * pi) ** 0.5))
        second = e ** (- ((x - self.mean) ** 2) / (2 * (self.stddev ** 2)))
        return first * second

    def cdf(self, x):
        """
        Calculate the cumulative distribution function (CDF) for a value x.

        Calcule la fonction de répartition cumulative (CDF) pour une valeur x
        selon la distribution normale.

        Args:
            x (float): Valeur pour laquelle calculer la probabilité cumulative.

        Returns:
            float: Probabilité cumulative de x.
        """
        def erf(x):
            """
            Approximation de la fonction d'erreur (erf) pour une valeur x.

            Args:
                x (float): Valeur pour laquelle calculer erf.

            Returns:
                float: Valeur approchée de erf(x).
            """
            return (2 / (pi ** 0.5)) * (
                x - ((x ** 3) / 3) + (
                    (x ** 5) / 10) + (
                    (x ** 7) / 42) + (
                    (x ** 9) / 216))

        return 0.5 * (1 + erf(x))
