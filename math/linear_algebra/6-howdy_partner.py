#!/usr/bin/env python3
"""
Return a new list that contain the content of arr1
followed by the content of arr2
"""


def cat_arrays(arr1, arr2):
    """cat_arrays

    Args:
        arr1 (list): list of ints
        arr2 (list): list of ints

    Returns:
        list: list of ints
    """
    new = arr1.copy()
    new.extend(arr2)
    return new
