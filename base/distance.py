#!/usr/bin/env python
# encoding: utf-8
"""
distance.py

Created by Julien Lauron on 2008-07-11.
Copyright (c) 2008 . All rights reserved.
"""

import scipy
from math import sqrt
from math import pow

def euclidian (pointa, pointb, dimension=2):
    res = 0

    for i in xrange (dimension):
        res += (pointa[i] - pointb[i]) * (pointa[i] - pointb[i])

    res = sqrt (res)

    return res

def manhattan (pointa, pointb, dimension=2):
    res = 0

    for i in xrange (dimension):
        res += abs (pointa[i] - pointb[i])

    return res

def minkowski (pointa, pointb, p, dimension=2):
    res = 0

    for i in xrange (dimension):
        res += pow (abs (pointa[i] - pointb[i]), p)

    res = pow (res, 1./p)

    return res

