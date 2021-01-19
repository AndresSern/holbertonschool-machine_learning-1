#!/usr/bin/env python3
"""
calculates the precision for each class in a confusion matrix:
"""
import numpy as np


def precision(confusion):
    """ precision for each class"""
    trp = np.diagonal(confusion)

    return trp / np.sum(confusion, axis=0)
