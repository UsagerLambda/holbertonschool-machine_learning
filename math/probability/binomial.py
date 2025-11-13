#!/usr/bin/env python3
"""Module pour la distribution Binomial."""

e = 2.7182818285
pi = 3.1415926536


class Binomial:
    """Classe représentant une distribution Binomial."""

    def __init__(self, data=None, n=1, p=0.5):
        """
        Initialise une instance de la distribution Binomiale.

        Si data est fourni,
            les paramètres n et p sont estimés à partir des données :
            - n est calculé comme la moyenne divisée par p, puis arrondi.
            - p est recalculé comme la moyenne divisée par n.
        Si data n'est pas fourni, les valeurs de n et p
            sont utilisées après validation.

        Args:
            data (list, optionnel): Liste de données à partir
                de laquelle estimé n et p.
            n (int, optionnel): Nombre d'essais (doit être > 0).
            p (float, optionnel): Probabilité de succès (0 < p < 1).

        Raises:
            ValueError: Si n <= 0, p <= 0, p >= 1,
                ou data contient moins de 2 valeurs.
            TypeError: Si data n'est pas une liste.
        """
        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")
            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
            self.n = int(n)
            self.p = float(p)
        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")

            if len(data) < 2:
                raise ValueError(
                    "data must contain multiple values")

            # μ = Σ(data) / len(data)
            mean = sum(data) / len(data)  # Calcul la moyenne

            # σ² = Σ(x - μ)² / len(data)
            variance = sum((x - mean) ** 2 for x in data) / len(data)

            # p = 1 - (σ² / μ)
            self.p = 1 - (variance / mean)

            # n = μ / p → puis arrondir
            self.n = round(mean / self.p)

            # p = μ / n (avec n arrondi)
            self.p = mean / self.n
