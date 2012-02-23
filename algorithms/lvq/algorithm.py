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

from algorithms.kohonen import algorithm as kohonen


class OLVQ (SomInterface):
    def __init__ (self, voronoi_vectors, dimension=2, iteration_max=None):
        if iteration_max is None:
            self.iteration_max = 50 * len(voronoi_vectors)
        else:
            self.iteration_max = iteration_max
        self._voronoi_vectors = voronoi_vectors

    def __str__ (self):
        return "OLVQ - %i Voronoi vectors" % len(self._voronoi_vectors)

    def draw_all (self):
        self._dataset.draw()
        for vect in self._voronoi_vectors:
            if vect.get_label() == 1:
                vect.draw_center(color="green")
            elif vect.get_label() == 0:
                vect.draw_center(color="black")
            else:
                vect.draw_center(color="blue")
        pylab.axis([-10, 20, -10, 20])
        pylab.show()

    def learn (self):
        data = self._dataset.data()
        labels = self._dataset.labels()
        if len(data) <= 0:
            return

        # Parameters
        alpha = 0.3

        changed_vectors = 1
        iter = 0
        while (iter < self.iteration_max) and (changed_vectors > 0):
            iter += 1

            random_point = int(random.random() * len(data))
            point = data[random_point]
            label = labels[random_point]

            # Get the  Voronoi vector
            _vector = self.__get_closest_voronoi_vector(point)

            if label == _vector.get_label():
                s = 1
            else:
                s = -1
            for i in xrange(len(_vector)):
                _vector[i] = (1 - s * alpha) * nearest_vector[i] + \
                    s * alpha * point[i]

            alpha = alpha / (1 + s * alpha)
            if alpha >= 1:
                alpha = 0.999999
            if alpha <= 0:
                alpha = 0.000001

        return iter

    def get_voronoi_vectors (self):
        return self._voronoi_vectors

    def __get_closest_voronoi_vector (self, point):
        """ Return the closest Voronoi vector for a given point """
        _vector = self._voronoi_vectors[0]
        best_distance = self._voronoi_vectors[0].distance_with(point)
        for i in xrange(1, len(self._voronoi_vectors)):
            tmp_distance = self._voronoi_vectors[i].distance_with(point)
            if tmp_distance < best_distance:
                _vector = self._voronoi_vectors[i]
                best_distance = tmp_distance

        return _vector

    def classify (self, point):
        """ Classify a given point by finding its closest Voronoi vector and
        returning the corresponding class """
        _vector = self.__get_closest_voronoi_vector(point)

        return _vector.get_label()

    def class_probability (self, point, c):
        """ Computes the probability that the  vector of a given point
        to have a given label. The probability is given as 0 < p < 1. This
        probability is independant of any treshold. """

        # We first get the closest Voronoi vector
        _vector = self.__get_closest_voronoi_vector(point)

        # We store every occurence of a label in a dictionary.
        frequences = {0: 0.0, 1: 0.0}
        for label in _vector._labels:
            if frequences.has_key(label):
                frequences[label] += 1.0
            else:
                frequences[label] = 0.0

        sum = 0
        for k in frequences:
            sum += frequences[k]

        if sum == 0:
            return 0
        else:
            return frequences[c] / sum

class LVQ (SomInterface):
    def __init__ (self, voronoi_vectors, dimension=2, iteration_max=None):
        if iteration_max is None:
            self.iteration_max = 50 * len(voronoi_vectors)
        else:
            self.iteration_max = iteration_max
        self._voronoi_vectors = voronoi_vectors

    def __str__ (self):
        return "LVQ - %i Voronoi vectors" % len(self._voronoi_vectors)

    def draw_all (self):
        self._dataset.draw()
        for vect in self._voronoi_vectors:
            if vect.get_label() == 1:
                vect.draw_center(color="green")
            elif vect.get_label() == 0:
                vect.draw_center(color="black")
            else:
                vect.draw_center(color="blue")
        pylab.axis([-10, 20, -10, 20])
        pylab.show()

    def learn (self):
        data = self._dataset.data()
        labels = self._dataset.labels()
        if len(data) <= 0:
            return

        # Parameters
        alpha = 0.1

        changed_vectors = 1
        iter = 0
        while (iter < self.iteration_max) and (changed_vectors > 0):
            iter += 1

            random_point = int(random.random() * len(data))
            point = data[random_point]
            label = labels[random_point]

            # Get the  Voronoi vector
            _vector = self.__get_closest_voronoi_vector(point)

            # Check class and update the vector accordingly
            if label == _vector.get_label():
                for i in xrange(len(_vector)):
                    _vector[i] = nearest_vector[i] + \
                      alpha * (point[i] - _vector[i])
            else:
                for i in xrange(len(_vector)):
                    _vector[i] = nearest_vector[i] - \
                      alpha * (point[i] - _vector[i])

            # Learning rate paramater update
            alpha -= 0.1 / self.iteration_max

        return iter

    def get_voronoi_vectors (self):
        return self._voronoi_vectors

    def __get_closest_voronoi_vector (self, point):
        """ Return the closest Voronoi vector for a given point """
        _vector = self._voronoi_vectors[0]
        best_distance = self._voronoi_vectors[0].distance_with(point)
        for i in xrange(1, len(self._voronoi_vectors)):
            tmp_distance = self._voronoi_vectors[i].distance_with(point)
            if tmp_distance < best_distance:
                _vector = self._voronoi_vectors[i]
                best_distance = tmp_distance

        return _vector

    def classify (self, point):
        """ Classify a given point by finding its closest Voronoi vector and
        returning the corresponding class """
        _vector = self.__get_closest_voronoi_vector(point)

        return _vector.get_label()

    def class_probability (self, point, c):
        """ Computes the probability that the  vector of a given point
        to have a given label. The probability is given as 0 < p < 1. This
        probability is independant of any treshold. """

        # We first get the closest Voronoi vector
        _vector = self.__get_closest_voronoi_vector(point)

        # We store every occurence of a label in a dictionary.
        frequences = {0: 0.0, 1: 0.0}
        for label in _vector._labels:
            if frequences.has_key(label):
                frequences[label] += 1.0
            else:
                frequences[label] = 0.0

        sum = 0
        for k in frequences:
            sum += frequences[k]

        if sum == 0:
            return 0
        else:
            return frequences[c] / sum


def test ():
    algorithm = kohonen.Kohonen([10, 10], 2, kohonen.gaussian_kernel, verbose=False)
    data = dataset.load("../../datasets/2gaussians1k.data")
    algorithm.load(data)
    algorithm.learn()
    #algorithm.draw_all()
    clusters = algorithm.get_clusters()
    lvq = LVQ(clusters, dimension=len(data[0]))
    lvq.load(data)
    lvq.learn()
    lvq.draw_all()

if __name__ == "__main__":
    test()
