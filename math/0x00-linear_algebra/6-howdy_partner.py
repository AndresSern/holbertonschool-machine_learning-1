#!/usr/bin/env python3
def cat_arrays(arr1, arr2):
    arr = []
    for i in range(len(arr1)):
        arr.append(arr1[i])
    for j in range(len(arr2)):
        arr.append(arr2[j])
    return arr
