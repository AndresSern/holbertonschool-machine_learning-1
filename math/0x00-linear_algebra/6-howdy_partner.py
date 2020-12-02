#!/usr/bin/env python3
""" cat_arrays"""


def cat_arrays(arr1, arr2):
    """ cat_arrays"""
    arr = []
    for i in range(len(arr1)):
        arr.append(arr1[i])
    for j in range(len(arr2)):
        arr.append(arr2[j])
    return arr
