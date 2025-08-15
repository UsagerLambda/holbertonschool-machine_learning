#!/usr/bin/env python3


def poly_integral(poly, C=0):
    if not isinstance(poly, list) or len(poly) == 0:
        return None
    if not isinstance(C, int):
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
