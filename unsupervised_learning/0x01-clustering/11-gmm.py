#!/usr/bin/env python3
"""
sklearn gmm
retrns:
weights_array-like of shape (n_components,)
The weights of each mixture components.

means_array-like of shape (n_components, n_features)
The mean of each mixture component.

covariances_array-like
The covariance of each mixture component.
"""
import sklearn.mixture


def gmm(X, k):
    """
    ARGS:
        -X is a numpy.ndarray of shape (n, d) containing the dataset
        -k is the number of clusters

    Returns: pi, m, S, clss, bic
        pi is a numpy.ndarray containing the cluster priors
        m is a numpy.ndarray containing the centroid means
        S is a numpy.ndarray containing the covariance matrices
        clss is a numpy.ndarray containing the cluster indices
            for each data point
        bic is a numpy.ndarray containing the BIC value for each
            cluster size tested
        """
    gmm = sklearn.mixture.GaussianMixture(k).fit(X)
    pi = gmm.weights_
    m = gmm.means_
    S = gmm.covariances_
    clss = gmm.predict(X)
    bic = gmm.bic(X)
    return pi, m, S, clss, bic
