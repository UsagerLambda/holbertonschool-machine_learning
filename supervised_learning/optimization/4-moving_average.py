#!/usr/bin/env python3
"""Calculate weighted moving average (with bias correction)."""


def moving_average(data, beta):
    """
    Calculate the weighted moving average of a data set.

    Args:
        data (list): list of data to calculate the moving average of
        beta (float): weight used for the moving average
    """
    moving_average = []
    avg = 0
    for i in range(len(data)):
        avg = beta * avg + (1 - beta) * data[i]  # EMA brut
        avg_corrected = avg / (1 - beta**(i + 1))  # EMA corrigé avec biais
        moving_average.append(avg_corrected)
    return moving_average
