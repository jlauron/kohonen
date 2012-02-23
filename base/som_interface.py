#!/usr/bin/env python
# encoding: utf-8
"""
SomInterface.py

Created by Julien Lauron on 2008-11-06
Copyright(c) All rights reserved.
"""

import math

class SomInterface:

    # Set the variable to True to see debug info.
    _verbose = False

    def __init__ (self):
        self._dataset = None

    def load (self, dataset):
        if len (dataset) == 0:
            raise Exception ()
        data = dataset.data()
        self._dataset = dataset

    def compute_distance(self, p1, p2):
        distance = 0.0
        for i in xrange(0, len(p2)):
            distance += (p1[i] - p2[i]) ** 2

        return math.sqrt(distance)

    def learn (self):
        pass

    def classify (self, point):
        pass

    def minimize_fun ():
        pass

    def affect_fun ():
        pass
