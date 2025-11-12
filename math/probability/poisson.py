#!/usr/bin/env python3
"""Module pour la distribution de Poisson."""

e = 2.7182818285
pi = 3.1415926536


class Poisson:
    """Classe représentant une distribution de Poisson."""

    def __init__(self, data=None, lambtha=1.):
        """
        Initialise la distribution de Poisson.

        Paramètres
        ----------
        data : list, optionnel
            Liste de données à utiliser pour estimer la distribution.
            Si None, lambtha est utilisé.
        lambtha : float, optionnel
            Nombre attendu d'occurrences dans un intervalle de temps donné
            (par défaut 1.).

        Comportement
        ------------
        - Définit l'attribut d'instance lambtha (toujours stocké comme float)
        - Si data n'est pas fourni (data is None) :
            - Utilise la valeur fournie de lambtha
            - Si lambtha n'est pas strictement positif, lève une ValueError
              avec le message :
                "lambtha must be a positive value"
        - Si data est fourni :
            - Calcule lambtha comme la moyenne des valeurs de data
            - Si data n'est pas une liste, lève une TypeError avec le message :
                "data must be a list"
            - Si data contient moins de deux valeurs, lève une ValueError
              avec le message :
                "data must contain multiple values"

        Exceptions
        ----------
        TypeError
            Si data est fourni et n'est pas une liste.
        ValueError
            Si data contient moins de deux valeurs.
        ValueError
            Si lambtha n'est pas strictement positif.
        """
        self.lambtha = float(lambtha)

        if data is None:
            if lambtha <= 0:
                raise ValueError(
                    "lambtha must be a positive value"
                )
            self.lambtha = float(lambtha)

        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")

            if len(data) < 2:
                raise ValueError(
                    "data must contain multiple values"
                )

            # sommes des données divisé par le nombre
            # des données pour récupérer la moyenne
            # lambda = moyenne
            self.lambtha = float(sum(data) / len(data))

    def pmf(self, k):
        """
        Calculate la valeur du PMF pour un nombre k de succès.

        Args:
            k (int): nombre de success

        Returns:
            float: valeur du PMF
        """
        # Vérifie que k est un entier
        k = int(k)

        # Vérifie si k est out of range
        if k < 0:
            return 0

        # Calcul la factorielle de k
        kfact = 1
        for i in range(1, k + 1):
            kfact *= i

        # Calculer la PMF: P(X = k) = (λ^k * e^(-λ)) / k!
        result = float(((e ** (-self.lambtha)) * (self.lambtha ** k)) / kfact)
        return result

    def cdf(self, k):
        """
        Calculate la value de CDF pour un nombre donné de succès.

        Args:
            k (int): nombre de succès

        Returns:
            float: valeur de CDF pour k succès
        """
        k = int(k)
        if k < 0:
            return 0
        result = 0

        for i in range(k+1):

            # Calcul la factoriel de i
            ifact = 1
            for j in range(1, i + 1):
                ifact *= j
            # -------------------------

            result += (e ** (-self.lambtha) * ((self.lambtha ** i) / ifact))

        return result
