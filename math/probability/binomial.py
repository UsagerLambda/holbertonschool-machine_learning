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

    def pmf(self, k):
        """
        Calculate la probabilité de masse (PMF) pour une valeur donnée k.

        La PMF d'une distribution binomiale donne la probabilité d'obtenir
        exactement k succès lors de n essais indépendants, chacun ayant une
        probabilité p de succès.

        Args:
            k (int): Nombre de succès pour lequel calculer la probabilité.

        Returns:
            float: Probabilité d'obtenir exactement k succès.

        Raises:
            ValueError: Si k n'est pas dans l'intervalle [0, n].
        """
        k = int(k)

        if k < 0:
            return 0

        if k > self.n:
            return 0

        def fact(x):
            """Calculate the factorial of x."""
            res = 1
            for i in range(1, x + 1):
                res *= i
            return res
        coef = fact(self.n) / (fact(k) * fact(self.n - k))
        return coef * (self.p ** k) * ((1 - self.p) ** (self.n - k))

    def cdf(self, k):
        """
        Calcule la fonction de répartition cumulative (CDF)
            pour une valeur donnée k.

        La CDF d'une distribution binomiale donne la probabilité d'obtenir
        au plus k succès lors de n essais indépendants, chacun ayant une
        probabilité p de succès.

        Args:
            k (int): Nombre de succès pour lequel calculer
                la probabilité cumulative.

        Returns:
            float: Probabilité d'obtenir au plus k succès.
        """
        k = int(k)

        if k < 0:
            return 0

        if k > self.n:
            return 0

        cdf_value = 0
        for i in range(0, k + 1):
            cdf_value += self.pmf(i)

        return cdf_value
