#!/usr/bin/env python
# encoding: utf-8
"""
main.py

Created by Julien Lauron on 2008-07-11.
Copyright (c) 2008 . All rights reserved.
"""

import sys
import getopt

import algorithm
from base import donut
from base import dataset
from base import crossvalidation as cv

help_message = '''
K Nearest Neighbour Classifier
'''

class Usage (Exception):
    def __init__ (self, msg):
        self.msg = msg


def test ():
    do = donut.Donut (1000, 0.1)
    datas, labels = do.get_toyproblem ()
    datas = dataset.Dataset (datas, labels)
    knn = algorithm.KNN (3)
    knn.load (datas[0:900])
    knn.learn ()
    test_dataset = datas[900:1000]
    data = test_dataset.data ()
    correct = 0
    for i in xrange (len (data)):
        label = knn.classify (data[i])
        if test_dataset.get_label (i) == label:
            correct += 1

    print "correct classification: %i / %i" % (correct, len(data))


def test_kfold ():
    #datas = dataset.load ("donut.data")
    do = donut.Donut (1000, 0.1)
    datas, labels = do.get_toyproblem ()
    datas = dataset.Dataset (datas, labels)
    kfold = cv.KFold (10)
    kfold.load_dataset (datas)
    kfold.estimate (algorithm.KNN (3))
    kfold.print_results ()


def main (argv=None):
    if argv is None:
        argv = sys.argv
    try:
        try:
            opts, args = getopt.getopt(argv[1:], "h", ["help"])
        except getopt.error, msg:
            raise Usage (msg)

        # option processing
        for option, value in opts:
            if option in ("-h", "--help"):
                raise Usage (help_message)

        test_kfold ()

    except Usage, err:
        print >> sys.stderr, str (err.msg)
        return 1


if __name__ == "__main__":
    sys.exit (main ())
