#!/usr/bin/env python3
"""sklearn kmeans"""
import sklearn.cluster


def kmeans(X, k):
    """
    class sklearn.cluster.KMeans(n_clusters=8, init=’k-means++’, n_init=10,
    max_iter=300, tol=0.0001, precompute_distances=’auto’, verbose=0,
    random_state=None, copy_x=True, n_jobs=1, algorithm=’auto’)
    Returns:

        *cluster_centers_ : array, [n_clusters, n_features]
            Coordinates of cluster centers

        *labels_ : :
            Labels of each point

        *inertia_ : float
            Sum of squared distances of samples
            to their closest cluster center.
    """
    kmeans = sklearn.cluster.KMeans(n_clusters=k).fit(X)
    return kmeans.cluster_centers_, kmeans.labels_
