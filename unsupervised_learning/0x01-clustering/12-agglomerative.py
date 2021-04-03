#!/usr/bin/env python3
""" performs agglomerative clustering on a dataset"""
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """ ARGS:
        -X is a numpy.ndarray of shape (n, d) containing the dataset
        -dist is the maximum cophenetic distance for all clusters
    Returns:
        - clss, a numpy.ndarray of shape (n,) containing the
        cluster indices for each data point
    """
    plt.figure(figsize=(10, 7))
    linkage = scipy.cluster.hierarchy.linkage(X, method='ward')
    dend = scipy.cluster.hierarchy.dendrogram(linkage, color_threshold=dist)
    clss = scipy.cluster.hierarchy.fcluster(Z=linkage, t=dist,
                                            criterion="distance")
    plt.show()
    return clss
