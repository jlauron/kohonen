#!/usr/bin/env python
# encoding: utf-8
"""
algorithm.py

Created by Julien Lauron on 2008-11-05
Copyright(c) All rights reserved.
"""

import scipy
import pylab
import random

from base import dataset
from base import distance

from base.som_interface import SomInterface
from base import cluster

class Kmeans(SomInterface):
    def __init__ (self, k, dimension=2, iteration_max=10000):
        self.k = int(k)
        self.iteration_max = iteration_max
        # Init clusters
        self._clusters = []
        for ncluster in xrange(self.k):
            new_cluster = cluster.Cluster(dimension=dimension)
            self._clusters.append(new_cluster)
        # Inherited from SomInterface:
        self._dataset = None

    def draw_all (self):
        self._dataset.draw()
        for cluster in self._clusters:
            cluster.draw_center()
        pylab.axis([-10, 20, -10, 20])
        pylab.show()

    def learn (self):
        data = self._dataset.data()
        if len(data) <= 0:
            return

        # Reinit clusters with points in the dataset
        for cluster in self._clusters:
            # Random point in dataset
            i = int(random.random() * len(data))
            cluster.set_center(data[i])

        changed_clusters = 1
        iter = 0
        while (iter < self.iteration_max) and (changed_clusters > 0):
            iter += 1

            for cluster in self._clusters:
                cluster.clear_points()

            # Affectation step
            for i in xrange(len(data)):
                point = data[i]

                # Get best cluster for current point
                best_cluster = self._clusters[0]
                best_distance = best_cluster.distance_with(point)

                for cluster in self._clusters:
                    tmp_distance = cluster.distance_with(point)
                    if tmp_distance < best_distance:
                        best_distance = tmp_distance
                        best_cluster = cluster

                best_cluster.add_point(point)

            # Minimization step
            changed_clusters = 0
            for cluster in self._clusters:
                changed_clusters += cluster.compute_average()

        return iter

    def get_clusters (self):
        return self._clusters

    def classify (self, point):
        best_cluster = self._clusters[0]
        best_distance = best_cluster.distance_with(point)

        for cluster in clusters:
            tmp_distance = cluster.distance_with(point)
            if tmp_distance < best_distance:
                best_distance = tmp_distance
                best_cluster = cluster

        return best_cluster


def test ():
    data = dataset.load("../../datasets/4gaussians1k.data")

    classifier = Kmeans(4)
    classifier.load(data)
    print "Kmeans finished in %i iterations over the dataset" % \
        classifier.learn()

    classifier.draw_all()
    pylab.show()

if __name__ == "__main__":
    test()
