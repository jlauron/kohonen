#!/usr/bin/env python
# encoding: utf-8
"""
cluster.py

Created by Julien Lauron on 2008-11-07
Copyright(c) All rights reserved.
"""

import distance
import pylab

class Cluster:
    def __init__ (self, dimension=2):
        self._center = []
        for j in xrange(dimension):
            self._center.append(0)
        self._points = []
        self._neurons = []
        self._labels = []
        self._dimension = dimension
        self._label = None

    def __setitem__ (self, key, item):
        self._center[key] = item

    def __getitem__ (self, key):
        return self._center[key]

    def __len__ (self):
        return len(self._center)

    def get_center (self):
        return self._center

    def set_center (self, center):
        self._center = center

    def add_point (self, point):
        self._points.append(point)

    def add_neuron(self, neuron):
        self._points.append(neuron.get_data())
        self._neurons.append(neuron)

    def add_label(self, label):
        self._labels.append(label)

    def clear_points (self):
        self._points = []

    def distance_with (self, point, distance_fun=distance.euclidian):
        result = distance_fun(self._center, point, dimension=self._dimension)

        return result

    def compute_average (self, distance_fun=distance.euclidian):
        """ Compute average point of all points affected to the current cluster,
        and move the center of the cluster to this new point """
        if len(self._points) <= 0:
            return 0

        if self._dimension != len(self._points[0]):
            raise Exception()

        # Initialize new center coords
        new_center = []
        for dim in xrange(self._dimension):
            new_center.append(0)

        # Compute average of all points coords
        for i in xrange(len(self._points)):
            for dim in xrange(self._dimension):
                new_center[dim] += self._points[i][dim]
        for dim in xrange(self._dimension):
            new_center[dim] = new_center[dim] / len(self._points)

        if self.distance_with(new_center) > 0:
            self._center = new_center
            return 1
        else:
            return 0

    def draw_center (self, color=None):
        """ Draw centers with pylab """
        if color == "green":
            pylab.plot([self._center[0]], [self._center[1]], "go")
        elif color == "black":
            pylab.plot([self._center[0]], [self._center[1]], "ko")
        elif color == "blue":
            pylab.plot([self._center[0]], [self._center[1]], "bo")
        else:
            pylab.plot([self._center[0]], [self._center[1]], "ro")

    def merge_with(self, other):
        self._points += other._points
        self._labels += other._labels
        self.compute_average()

    def get_label (self):
        """ Get the cluster main label """

        most_frequent = None

        frequences = {}
        most_frequent = None
        for label in self._labels:
            if frequences.has_key(label):
                frequences[label] += 1
            else:
                frequences[label] = 0

            freq = None
            for label in frequences:
                if most_frequent is None:
                    most_frequent = label
                    freq = frequences[label]
                else:
                    if frequences[label] > freq:
                        most_frequent = label
                        freq = frequences[label]

        return most_frequent
