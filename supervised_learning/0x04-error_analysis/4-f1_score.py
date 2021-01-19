#!/usr/bin/env python3
"""
calculates the F1 score of a confusion matrix:
"""

import numpy as np


def f1_score(confusion):
    """
    F1 score
    confusion is a confusion numpy.ndarray of shape (classes, classes)
    classes is the number of classes
    """
    FP = confusion.sum(axis=0) - np.diag(confusion)
    FN = confusion.sum(axis=1) - np.diag(confusion)
    TP = np.diag(confusion)
    F1 = TP / (TP + (0.5 * (FN + FP)))
    return F1
