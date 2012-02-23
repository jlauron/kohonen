#!/usr/bin/env python
# encoding: utf-8
"""
algorithm.py

Created by Julien Lauron on 2008-07-11.
Copyright (c) 2008 . All rights reserved.
"""

import scipy
from PyML import *

from base import dataset as d

class SVM:
    def __init__ (self, C, kernel=None):
        if kernel is None:
            self.svm = svm.SVM (C=C)
        else:
            self.svm = svm.SVM (kernel, C=C)

    def load (self, dataset):
        if len (dataset) == 0:
            raise Exception ()
        self.__data = dataset.data ()
        self.__labels = dataset.labels ()
        self.data = VectorDataSet(self.__data, L=self.__labels)

    def learn (self):
        self.svm.train (self.data)

    def kfold (self, k):
        result = self.svm.cv (self.data, num_folds=k)

        return result

    def classify (self, dataset):
        results = self.svm.test (dataset)

        return results

    def draw_roc (self):
        pass

