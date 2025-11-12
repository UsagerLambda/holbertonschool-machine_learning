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
        Calcule le score z d'une valeur x.

        Args:
            x (float): Valeur à transformer en score z.

        Returns:
            float: Score z correspondant à x.
        """
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """
        Calcule la valeur x à partir d'un score z.

        Args:
            z (float): Score z à transformer en valeur x.

        Returns:
            float: Valeur x correspondant au score z.
        """
        return z * self.stddev + self.mean
