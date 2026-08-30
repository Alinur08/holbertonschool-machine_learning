#!/usr/bin/env python3
"""Agglomerative clustering using scipy"""
import scipy.cluster.hierarchy
import matplotlib.pyplot as plt


def agglomerative(X, dist):
    """
    Performs agglomerative clustering on a dataset with Ward linkage
    """
    links = scipy.cluster.hierarchy.linkage(X, method='ward')

    scipy.cluster.hierarchy.dendrogram(links, color_threshold=dist)
    plt.show()

    clss = scipy.cluster.hierarchy.fcluster(links, t=dist,
                                             criterion='distance')

    return clss
