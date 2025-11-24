#!/usr/bin/env python3
"""Calculate the likelihood of obtaining side effects by taking a new drug."""

import numpy as np


def likelihood(x, n, P):
    """
    Calculate the likelihood of obtaining side effects by taking a new drug.

    Args:
        x (int): number of patients that develop severe side effects
        n (int): total number of patients observed
        P (numpy.ndarray): 1D numpy.ndarray containing the various
        hypothetical probabilities of developing severe side effects.

    Raises:
        ValueError: n must be a positive integer
        ValueError: x must be an integer that is greater than or equal to 0
        ValueError: x cannot be greater than n
        TypeError: P must be a 1D numpy.ndarray
        ValueError: All values in P must be in the range [0, 1]

    Return:
        1D numpy.ndarray containing the likelihood of obtaining the data,
        x and n, for each probability in P, respectively
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(P, np.ndarray) or P.ndim > 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if np.any((P < 0) | (P > 1)):
        raise ValueError("All values in P must be in the range [0, 1]")

    def fact(x):
        """Calculate the factorial of x."""
        res = 1
        for i in range(1, x + 1):
            res *= i
        return res

    # C(n, x) = n! / (x! × (n-x)!)
    # Nombre de combinaisons de x cas parmi n patients
    coef = fact(n) / (fact(x) * fact(n - x))

    likelihood = coef * (P ** x) * ((1 - P) ** (n - x))

    return likelihood


def intersection(x, n, P, Pr):
    """
    Calculate something.

    Calculate the intersection of obtaining this data with
    the various hypothetical probabilities.

    Args:
        x (int): _description_
        n (int): _description_
        P (np.ndarray): 1D np.ndarray containing the various
        Pr (np.ndarray): 1D np.ndarray containing the prior beliefs op P (P(A))
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(P, np.ndarray) or P.ndim > 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")

    if np.any((P < 0) | (P > 1)):
        raise ValueError("All values in P must be in the range [0, 1]")

    if np.any((Pr < 0) | (Pr > 1)):
        raise ValueError("All values in Pr must be in the range [0, 1]")

    if not np.isclose(sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    # P(A ∩ B) = P(A) × P(B)
    return Pr * likelihood(x, n, P)


def marginal(x, n, P, Pr):
    """
    Calculate the calculates the marginal probability of obtaining the data.

    Args:
        x (int): _description_
        n (int): _description_
        P (np.ndarray): 1D np.ndarray containing the various
        Pr (np.ndarray): 1D np.ndarray containing the prior beliefs op P (P(A))
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(P, np.ndarray) or P.ndim > 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")

    if np.any((P < 0) | (P > 1)):
        raise ValueError("All values in P must be in the range [0, 1]")

    if np.any((Pr < 0) | (Pr > 1)):
        raise ValueError("All values in Pr must be in the range [0, 1]")

    if not np.isclose(sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    # P(B) = Σ P(B | Aᵢ) × P(Aᵢ)
    return np.sum(likelihood(x, n, P) * Pr)


def posterior(x, n, P, Pr):
    """
    Calculate something.

    Calculate the calculates the posterior probability
    for the various hypothetical probabilities of developing
    severe side effects given the data.

    Args:
        x (int): _description_
        n (int): _description_
        P (np.ndarray): 1D np.ndarray containing the various
        Pr (np.ndarray): 1D np.ndarray containing the prior beliefs op P (P(A))
    """
    if not isinstance(n, int) or n <= 0:
        raise ValueError("n must be a positive integer")

    if not isinstance(x, int) or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0")

    if x > n:
        raise ValueError("x cannot be greater than n")

    if not isinstance(P, np.ndarray) or P.ndim > 1:
        raise TypeError("P must be a 1D numpy.ndarray")

    if not isinstance(Pr, np.ndarray) or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")

    if np.any((P < 0) | (P > 1)):
        raise ValueError("All values in P must be in the range [0, 1]")

    if np.any((Pr < 0) | (Pr > 1)):
        raise ValueError("All values in Pr must be in the range [0, 1]")

    if not np.isclose(sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    # P(A|B) = P(B | Aᵢ) * P(Aᵢ) / Σ P(B | Aᵢ) * P(Aᵢ)
    return (likelihood(x, n, P) * Pr) / np.sum(likelihood(x, n, P) * Pr)
