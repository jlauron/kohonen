#!/usr/bin/env python
# encoding: utf-8
"""
donut.py

Created by Julien Lauron on 2008-07-11.
Copyright (c) 2008 . All rights reserved.
"""

import scipy
import random
import pylab

import distance

## CONSTANTS ##

MAX_X = 10
MIN_X = -10
MAX_Y = 10
MIN_Y = -10

FIRST_RADIUS = 3.5
SECOND_RADIUS = 8.5
CENTER = (0, 0)

class Donut:
    def __init__ (self, n, noise=0):
        self.n = n
        self.noise = noise
        self.__data = scipy.zeros ((n, 2))
        self.__labels = []

        for i in xrange (n):
            coord_x = random.random ()
            coord_x = MIN_X + (MAX_X - MIN_X) * coord_x
            coord_y = random.random ()
            coord_y = MIN_Y + (MAX_Y - MIN_Y) * coord_y

            self.__data[i, 0] = coord_x
            self.__data[i, 1] = coord_y

            self.__labels.append (self.compute_label (self.__data[i], noise))

    def get_toyproblem (self):
        return self.__data, self.__labels

    def compute_label (self, point, noise):
        if (distance.euclidian (CENTER, point) <= FIRST_RADIUS) or \
          (distance.euclidian (CENTER, point) >= SECOND_RADIUS):
            proba = random.random ()
            if proba < noise:
                return "in"
            else:
                return "out"
        else:
            proba = random.random ()
            return "in"

    def get_label (self, point):
        return self.__labels[point]

    def draw (self):
        for i in xrange (self.n):
            if self.__labels[i] == "in":
                pylab.plot ([self.__data[i, 0]], [self.__data[i, 1]], "r^")
            else:
                pylab.plot ([self.__data[i, 0]], [self.__data[i, 1]], "bo")

        pylab.axis ([MIN_X, MAX_X, MIN_Y, MAX_Y])
        pylab.show ()
