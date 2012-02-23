#!/usr/bin/env python
# encoding: utf-8
"""
dataset.py

Created by Julien Lauron on 2008-07-11.
Copyright (c) 2008 . All rights reserved.
"""

import scipy
import c_pickle
import pylab
import random

NRANDOM = 1000

def load (filename):
    file = open (filename, "rb")
    dataset = c_pickle.load (file)
    file.close ()
    return dataset

def save (dataset, filename):
    output = open (filename, "wb")
    c_pickle.dump (dataset, output)
    output.close ()

def import_dataset(filename, label_first = True):
    data = []
    labels = []
    file = open(filename, "r")

    while file:
        line = file.readline().split()

        if len(line) == 0:
            break

        for i in xrange(len(line)):
            line[i] = eval(line[i])

        if label_first:
            labels.append(line[0])
            data.append(line[1:])
        else:
            labels.append(line[len(line) - 1])
            data.append(line[:len(line) - 1])

    return Dataset(data, labels)

class Dataset:
    def __init__ (self, data, labels):
        self.__data = scipy.array (data)
        self.__labels = labels

    def __len__ (self):
        return len (self.__data)

    def __repr__ (self):
        return self.__str__ ()

    def __str__ (self):
        return str (self.__data)

    def __getitem__ (self, key):
        return Dataset (self.__data[key], self.__labels[key])

    def data (self):
        return self.__data

    def labels (self):
        return self.__labels

    def get_label (self, point):
        return self.__labels[point]

    def draw (self):
        if len (self.__labels) > 0:
            for i in xrange (len (self.__data)):
                label = "bo"
                if self.__labels[i] == "in":
                    label = "r^"
                elif self.__labels[i] == 0:
                    label = "r+"
                elif self.__labels[i] == 1:
                    label = "y+"
                elif self.__labels[i] == 2:
                    label = "g+"
                elif self.__labels[i] == 3:
                    label = "b+"
                pylab.plot ([self.__data[i, 0]], [self.__data[i, 1]], label)
        else:
            pylab.plot (self.__data[:,0], self.__data[:,1], "b+")

    def show (self):
        pylab.axis ([-20, 10, -20, 10])
        pylab.show ()

    def random (self):
        data = self.__data
        labels = self.__labels

        for i in xrange (NRANDOM):
            first = random.randint (0, len (data) - 1)
            second = random.randint (0, len (data) - 1)

            tmp = data[first].copy ()
            data[first] = data[second].copy ()
            data[second] = tmp
            if len(labels) > 0:
                tmp_label = labels[first]
                labels[first] = labels[second]
                labels[second] = tmp_label

        self.__data = data
        self.__labels = labels

