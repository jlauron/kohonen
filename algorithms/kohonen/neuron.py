#!/usr/bin/env python
# encoding: utf-8
"""
neuron.py

Created by Julien Lauron on 2008-11-08
Copyright(c) All rights reserved.
"""

import pylab

class Neuron:
    def __init__ (self, position=[], data=[]):
        self._position = position
        self._data = data
        self._labels = []
        self._cluster = None
        self._context = None

    def __getitem__ (self, key):
        return self._data[key]

    def __setitem__ (self, key, value):
        self._data[key] = value

    def __repr__ (self):
        return str(self._data)

    def __len__ (self):
        return len(self._data)

    def set_data (self, data):
        self._data = data

    def get_data (self):
        return self._data

    def set_position (self, position):
        self._position = position

    def get_context(self):
        return self._context

    def set_context(self, context):
        self._context = context

    def get_position (self):
        return self._position

    def draw (self):
        if self._cluster is None:
            pylab.plot([self._data[0]], [self._data[1]], "ro")
        else:
            clusters = {None : "wo", 0 : "bo", 1 : "ro"}
            pylab.plot([self._data[0]], [self._data[1]], clusters[self._cluster])

    def class_probability(self, c):
        frequences = {0: 0.0, 1: 0.0}
        for label in self._labels:
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
