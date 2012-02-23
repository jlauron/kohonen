#!/usr/bin/env python
# encoding: utf-8
"""
algorithm.py

Created by Julien Lauron on 2008-11-08
Copyright(c) All rights reserved.
"""

import scipy
import pylab
import math
import random
import copy

from base.som_interface import SomInterface
from base import dataset
from base import distance
from base import cluster

from neuron import Neuron

ITERATION_MAX = 5000
PERC_NEURONS_CHANGED = 5

def generate_positions (dimensions, acc=[]):
    """ Build indexes from list of dimensions """
    if len(dimensions) == 0:
        yield acc
    else:
        index = dimensions.pop(0)
        for i in xrange(index):
            mydimensions = acc[:]
            mydimensions.append(i)
            for result in generate_positions(dimensions[:], mydimensions):
                yield result

def neuron_with_position (neurons, position):
    for neuron in neurons:
        neuron_pos = neuron.get_position()
        if neuron_pos == position:
            return neuron

    return None

def next_neurons (neurons, position):
    for i in xrange(len(position)):
        next_position = position[:]
        next_position[i] = next_position[i] + 1
        next_neuron = neuron_with_position(neurons, next_position)
        yield next_neuron

def gaussian_kernel (distance, sigma):
    return math.exp(-distance * distance / (2 * sigma * sigma))

def clusterize (neurons, data_dimension, clusters, dataset, nb_clusters=0):
    # Each neuron is put in its own cluster.
    for neuron in neurons:
        c = cluster.Cluster(data_dimension)
        c.add_neuron(neuron)
        c.compute_average()
        clusters.append(c)

    # Every element from the dataset is classified and the corresponding
    # cluster stores its label.
    for z in dataset:
        nearest_cluster = clusters[0]
        best_distance = distance.euclidian(clusters[0], z.data())
        for i in xrange(1, len(clusters)):
            tmp_distance = distance.euclidian(clusters[i], z.data())
            if tmp_distance < best_distance:
                nearest_cluster = clusters[i]
                best_distance = tmp_distance
        nearest_cluster.add_label(z.labels())
        nearest_cluster._neurons[0]._labels.append(z.labels())

    # Here we reduce the number of cluster to the one specified. To do
    # that we look for the two closest clusters and we merge them until
    #  we have the good number of clusters.
    if nb_clusters > 0:
        while len(clusters) > nb_clusters:
            best_distance = clusters[0].distance_with(clusters[1].get_center())
            cluster1 = 0
            cluster2 = 1
            for i in xrange(len(clusters)):
                for j in xrange(len(clusters)):
                    if i != j:
                        tmp_distance = clusters[i].distance_with(clusters[j].get_center())
                        if tmp_distance < best_distance:
                            cluster1 = i
                            cluster2 = j
                            best_distance = tmp_distance
            clusters[i].merge_with(clusters[j])
            clusters.pop(j)

def get_nearest_cluster (clusters, point):
    """ Retrieves the closest cluster of a point. """

    # Initially the result is set to the first neuron.
    nearest_cluster = clusters[0]
    best_distance = distance.euclidian(clusters[0], point)

    # We look for a closer neuron in the neuron list.
    for i in xrange(1, len(clusters)):
        tmp_distance = distance.euclidian(clusters[i], point)
        if tmp_distance < best_distance:
            nearest_cluster = clusters[i]
            best_distance = tmp_distance

    return nearest_cluster

class Kohonen (SomInterface):
    def __init__ (self, map_dimensions, data_dimension, kernel_func, nb_clusters=0, verbose=False):
        self._verbose = verbose
        self._map_dimensions = []
        self._map_dimensions[:] = map_dimensions[:]
        self._data_dimension = data_dimension
        self._dataset = None
        self._kernel = kernel_func
        self._nb_clusters = nb_clusters
        self._neurons = []
        self._clusters = []

        for position in generate_positions(map_dimensions):
            # Initialize _neurons positions in map
            neuron = Neuron(position=position)
            # Initialize _neurons coordinates
            coords = []
            for i in xrange(data_dimension):
                coords.append(random.random() * 10)
            neuron.set_data(coords)
            self._neurons.append(neuron)

    def __str__(self):
        res = "Kohonen"
        res += ", dimensions: " + str(self._map_dimensions)
        if self._nb_clusters > 0:
            res += ", clusters: " + str(self._nb_clusters)

        return res

    def draw_map (self):
        for neuron in self._neurons:
            position = neuron.get_position()
            if len(position) <= 2:
                for next_neuron in next_neurons(self._neurons, position):
                    if not next_neuron is None:
                        pylab.plot([neuron[0], next_neuron[0]],
                                [neuron[1], next_neuron[1]],
                                "r-")
            neuron.draw()

    def draw_dataset (self):
        self._dataset.draw()

    def draw_all (self):
        self._dataset.draw()
        self.draw_map()
        pylab.axis([-10, 20, -10, 20])
        pylab.show()

    def draw_classifier(self):
        width = self._map_dimensions[0]
        height = self._map_dimensions[1]

        x = scipy.arange(0, width, 1)
        y = scipy.arange(0, height, 1)

        map = []
        for i in x:
            line = []
            for j in y:
                line.append(self._neurons[j * width + i].class_probability(1))
            map.append(line)

        pylab.imshow(map, interpolation='bilinear')
        pylab.grid(True)

    def learn (self):
        if self._verbose:
            self.draw_all()
        # Parameters initialization
        neurons = self._neurons
        sigma_init = distance.euclidian(neurons[len(neurons) / 2].get_position(), \
                neurons[len(neurons) - 1].get_position(), \
                dimension=len(neurons[0].get_position()))
        t1 = 1000 / math.log(sigma_init)
        nu0 = 0.1
        t2 = 1000

        nu = lambda n: nu0 * math.exp(-n / t2)

        iter = 1
        data = self._dataset.data()

        while (iter <= ITERATION_MAX):
            z = data[int(random.random() * len(data))]

            sigma = sigma_init * math.exp(-iter / t1)

            # Get nearest neuron for observation z
            nearest_neuron = neurons[0]
            best_distance = distance.euclidian(neurons[0], z)
            for i in xrange(1, len(neurons)):
                tmp_distance = distance.euclidian(neurons[i], z)
                if tmp_distance < best_distance:
                    nearest_neuron = neurons[i]
                    best_distance = tmp_distance

            if (iter % len(neurons) == 0):
                changed_neurons = 0
            else:
                changed_neurons = None
            for neuron in neurons:
                d = distance.euclidian(neuron.get_position(), \
                        nearest_neuron.get_position(),         \
                        dimension=len(neuron.get_position()))
                h = gaussian_kernel(d, sigma)
                dmoved = 0
                for i in xrange(len(neuron)):
                    step = nu(iter) * h * (z[i] - neuron[i])
                    neuron[i] += step
                    dmoved += abs(step)

                if not changed_neurons is None:
                    if dmoved > 0:
                        changed_neurons += 1

            if not changed_neurons is None:
                perc = float(changed_neurons) / len(neurons) * 100
                if perc <= PERC_NEURONS_CHANGED:
                    break

            if iter % 1000 == 0 and self._verbose:
                print "%f%% of neurons moved at last check" % perc
                print "iteration %i" % iter
                self.draw_all()

            iter += 1

        if self._verbose:
            print "%i%% of neurons moved at last check" % int(perc)
            self.draw_all()

        clusterize(self._neurons, self._data_dimension, self._clusters, self._dataset, self._nb_clusters)

        return iter - 1

    def get_nearest_cluster(self, point):
        """ Retrieves the closest cluster of a point. """

        # Initially the result is set to the first neuron.
        nearest_cluster = self._clusters[0]
        best_distance = distance.euclidian(self._clusters[0], point)

        # We look for a closer neuron in the neuron list.
        for i in xrange(1, len(self._clusters)):
            tmp_distance = distance.euclidian(self._clusters[i], point)
            if tmp_distance < best_distance:
                nearest_cluster = self._clusters[i]
                best_distance = tmp_distance

        return nearest_cluster

    def class_probability(self, point, c):
        """ Computes the probability that the nearest cluster of a given point
        to have a given label. The probability is given as 0 < p < 1. This
        probability is independant of any treshold. """

        # We first get the closest cluster.
        nearest_cluster = get_nearest_cluster(self._clusters, point)

        # We store every occurence of a label in a dictionary.
        frequences = {0: 0.0, 1: 0.0}
        for label in nearest_cluster._labels:
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

    def classify (self, point):
        """ Computes the guessed label of a given point. """

        nearest_cluster = get_nearest_cluster(self._clusters, point)

        return nearest_cluster.get_label()

    def get_clusters (self):
        return self._clusters

def test ():
    algorithm = Kohonen([10, 10], 2, gaussian_kernel, verbose=False)
    data = dataset.load("../../datasets/4gaussians1k.data")
    algorithm.load(data)
    print "Kohonen finished in %i iterations" % algorithm.learn()
    print algorithm.classify(data.data()[0])
    print algorithm.class_probability(data.data()[0], 1)
    algorithm.draw_all()

if __name__ == "__main__":
    test()
