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
            if stddev < 0:
                raise ValueError("stddev must be a positive value")

        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")

            if len(data) < 2:
                raise ValueError(
                    "data must contain multiple values")

            self.mean = float(sum(data) / len(data))

            variance = sum((x - self.mean) ** 2 for x in data) / (
                len(data) - 1)
            self.stddev = float(variance ** 0.5)
