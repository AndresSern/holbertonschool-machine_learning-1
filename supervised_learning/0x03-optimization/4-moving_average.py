#!/usr/bin/env python3
"""
calculates the weighted
moving average of a data set:
"""


def moving_average(data, beta):
    """
    calculates the weighted
    moving average of a data set:
    """
    newdata = []
    v = 0
    for i in range(0, len(data)):
        v = v * beta + ((1 - beta) * data[i])
        newdata.append(v / (1 - beta**(i+1)))
    return newdata
