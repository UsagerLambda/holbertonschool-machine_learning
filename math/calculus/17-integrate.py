#!/usr/bin/env python3
"""
calculates the integral of a polynomial
"""


def poly_integral(poly, C=0):
    """
    Args:
        poly (list): List of integer
        C (int, optional): integer representing the integration constant.
        Defaults to 0.

    Returns:
        _type_: new list of coefficients representing
        the integral of the polynomial
    """
    if not isinstance(poly, list) or len(poly) == 0:
        return None
    if not isinstance(C, int):
        return None
    if not (isinstance(elem, int) for elem in poly):
        return None

    integral = [C]
    for i in range(len(poly)):
        coeff = poly[i]
        diviseur = i + 1
        if coeff % diviseur == 0:
            integral.append(coeff // diviseur)
        else:
            integral.append(coeff / diviseur)
    return integral
