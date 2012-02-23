#!/usr/bin/env python
# encoding: utf-8
"""
algorithm.py

Created by Julien Lauron on 2008-07-11.
Copyright (c) 2008 . All rights reserved.
"""

import scipy

from base import dataset as d
from base import distance

def dist_cmp (x, y):
    first_dist = x[0]
    second_dist = y[0]
    if first_dist > second_dist:
        return 1
    if first_dist == second_dist:
        return 0
    return -1

class KNN:
    def __init__ (self, k):
        self.k = k

    def __str__(self):
        return "KNN, k: " + str(self.k)

    def load (self, dataset):
        if len (dataset) == 0:
            raise Exception ()
        self.__data = dataset.data ()
        self.__labels = dataset.labels ()

    def learn (self):
        pass

    def classify (self, point):
        distances = []

        # Compute distances with all dataset
        for i in xrange (len (self.__data)):
            dist = distance.euclidian (self.__data[i], point)
            distances.append ((dist, self.__labels[i]))

        # Keep only K nearest points
        distances.sort (dist_cmp)
        distances = distances[:self.k]
        # Compute the most frequent label in neighbours
        labels_freq = {}
        for i in xrange (len (distances)):
            if not labels_freq.has_key (distances[i][1]):
                labels_freq[distances[i][1]] = 0
            else:
                labels_freq[distances[i][1]] += 1
        i = 0
        for label in labels_freq:
            if i == 0:
                label_max = label
                freq_max = labels_freq[label]
                i += 1
            else:
                if labels_freq[label] > freq_max:
                    label_max = label
                    freq_max = labels_freq[label]

        return label_max

    def class_probability (self, point, c):
        distances = []

        # Compute distances with all dataset
        for i in xrange (len (self.__data)):
            dist = distance.euclidian (self.__data[i], point)
            distances.append ((dist, self.__labels[i]))

        # Keep only K nearest points
        distances.sort (dist_cmp)
        distances = distances[:self.k]
        # Compute the most frequent label in neighbours
        labels_freq = {}
        freq_tot = 0.0
        ref_freq = 0.0
        for i in xrange (len (distances)):
            freq_tot += 1
            if distances[i][1] == c:
                ref_freq += 1
            if not labels_freq.has_key (distances[i][1]):
                labels_freq[distances[i][1]] = 0
            else:
                labels_freq[distances[i][1]] += 1
        i = 0
        for label in labels_freq:
            if i == 0:
                label_max = label
                freq_max = labels_freq[label]
                i += 1
            else:
                if labels_freq[label] > freq_max:
                    label_max = label
                    freq_max = labels_freq[label]

        return ref_freq / freq_tot
