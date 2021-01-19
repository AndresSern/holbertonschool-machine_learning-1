#!/usr/bin/env python3
"""  creates a confusion matrix:"""

import numpy as np


def create_confusion_matrix(labels, logits):
    """  creates a confusion matrix:"""
    m, classes = logits.shape
    confusion_matrix = np.zeros((classes, classes))
    for i in range(m):
        confusion_matrix[np.where(labels[i, :] == 1),
                         np.where(logits[i, :] == 1)] += 1
    return confusion_matrix
