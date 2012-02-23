#!/usr/bin/env python
#a encoding: utf-8
"""
crossvalidation.py

Created by Julien Lauron on 2008-07-11.
Copyright (c) 2008 . All rights reserved.
"""

from dataset import Dataset
import scipy

import math
import random

NRANDOM = 0

class KFold:
    def __init__ (self, k):
        self.k = k

    def estimate (self, algorithm):
        self.results = []
        subds_len = len (self.dataset) / self.k

        for i in xrange (self.k):
            offset = i * subds_len
            # Build test dataset
            if i == (self.k - 1):
                testset = self.dataset[offset:len(self.dataset) - 1]
            else:
                testset = self.dataset[offset:offset + subds_len]

            # Build train dataset
            indexes = []
            # build data
            for index in xrange (len (self.dataset)):
                if index < offset or index >= (offset + subds_len):
                    indexes.append (index)
            trainset_data = scipy.take (self.dataset.data (), indexes, axis=0)
            # build labels for data
            labels = self.dataset.labels ()
            trainset_labels = []
            for index in indexes:
                trainset_labels.append (labels[index])
            trainset = Dataset (trainset_data, trainset_labels)

            # Test algorithm
            algorithm.load (trainset)
            algorithm.learn ()
            correct = 0
            testset_data = testset.data ()
            for test in xrange (len (testset)):
                label = algorithm.classify (testset_data[test])
                if testset.get_label (test) == label:
                    correct += 1

            self.results.append ((float (correct) / len (testset)) * 100)

        mean = 0
        for res in self.results:
            mean += res
        mean /= len (self.results)

        return mean

    def load_dataset (self, dataset):
        data = dataset.data ()
        labels = dataset.labels ()

        for i in xrange (NRANDOM):
            first = random.randint (0, len (data) - 1)
            second = random.randint (0, len (data) - 1)

            tmp = data[first].copy ()
            tmp_label = labels[first]
            data[first] = data[second].copy ()
            labels[first] = labels[second]
            data[second] = tmp
            labels[second] = tmp_label

        self.dataset = Dataset (data, labels)

    def print_results (self):
        mean = 0
        for res in self.results:
            mean += res
        mean /= len (self.results)
        print "Results for %i-Fold cross validation:" % self.k
        for i in xrange (len (self.results)):
            print "%i - %f %%" % (i + 1, self.results[i])
        print "Mean: %f %%" % mean

    def get_expected_value(self):
        """ Computes the expected value of the last cross validation. This
        value is computed as sum(x_i * p_i). """

        mean = 0.0
        for res in self.results:
            mean += res
        mean /= len(self.results)

        return mean

    def get_standard_deviation(self):
        """ Computes the standard deviation of the last cross validation. This
        value is computed as sqrt(1/N * sum((x_i - mean(x_i)) ^ 2)). """

        mean = self.get_expected_value()

        standard_deviation = 0.0
        for res in self.results:
            standard_deviation += (res - mean) ** 2
        standard_deviation /= len(self.results)

        return math.sqrt(standard_deviation)
