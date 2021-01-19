#!/usr/bin/env python3
"""
calculates the sensitivity for each class in a confusion matrix:
"""
import numpy as np


def sensitivity(confusion):
    """ sensitivity for each class"""
    trp = np.diagonal(confusion)
    return trp / np.sum(confusion, axis=1)
