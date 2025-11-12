#!/usr/bin/env python3
"""Module pour la distribution de Exponential."""

e = 2.7182818285
pi = 3.1415926536


class Exponential:
    """Classe représentant une distribution de Exponentials."""

    def __init__(self, data=None, lambtha=1.):
        """
        Initialise la distribution de Exponentials.

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
            - Calcule lambtha comme l'inverse de la moyenne des valeurs de data
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
                    "lambtha must be a positive value")
            self.lambtha = float(lambtha)

        else:
            if not isinstance(data, list):
                raise TypeError("data must be a list")

            if len(data) < 2:
                raise ValueError(
                    "data must contain multiple values")

            self.lambtha = float(1 / (sum(data) / len(data)))
