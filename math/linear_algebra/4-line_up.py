#!/usr/bin/env python3
"""
Function that return the sum of the i index of two array in a new one
"""


def add_arrays(arr1, arr2):
    """add_arrays

    Args:
        arr1 (list): list of ints
        arr2 (list): list of ints

    Returns:
        list: sum of element at i index of each lists
    """
    new = []
    if len(arr1) == len(arr2):
        for i in range(len(arr1)):
            new.append(arr1[i] + arr2[i])
        return new
    else:
        return None
