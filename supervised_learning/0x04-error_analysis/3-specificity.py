#!/usr/bin/env python3
"""
calculates the specificity for each class in a confusion matrix:
"""
import numpy as np


def specificity(confusion):
    """
    specificity
    confusion is a confusion numpy.ndarray of shape (classes, classes)
    classes is the number of classes
    """
    FP = confusion.sum(axis=0) - np.diag(confusion)
    FN = confusion.sum(axis=1) - np.diag(confusion)
    TP = np.diag(confusion)
    TN = confusion.sum() - (FP + FN + TP)
    TNR = TN/(TN+FP)
    return TNR
