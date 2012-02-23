#!/usr/bin/env python
# encoding: utf-8
"""
kfold.py

Created by Julien Lauron on @DATE@
Copyright(c) All rights reserved.
"""

import algorithm as lvq
from algorithms.kohonen import algorithm as kohonen
from base.crossvalidation import *
from base import dataset

if __name__ == "__main__":
    data = dataset.load("../../datasets/2gaussians1k.data")

    algorithm = kohonen.Kohonen([10, 10], 2, kohonen.gaussian_kernel, verbose=False)
    algorithm.load(data)
    algorithm.learn()

    kfold = KFold(10)
    kfold.load_dataset(data)
    print "KFold on LVQ"
    lvqalgo = lvq.LVQ(algorithm.get_clusters())
    kfold.estimate(lvqalgo)
    kfold.print_results()

    algorithm = kohonen.Kohonen([10, 10], 2, kohonen.gaussian_kernel, verbose=False)
    algorithm.load(data)
    algorithm.learn()

    kfold = KFold(10)
    kfold.load_dataset(data)
    print "KFold on OLVQ"
    lvqalgo = lvq.OLVQ(algorithm.get_clusters())
    kfold.estimate(lvqalgo)
    kfold.print_results()
