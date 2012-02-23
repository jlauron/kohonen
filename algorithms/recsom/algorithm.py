import random
import math

from algorithms.kohonen import algorithm as som
from algorithms.kohonen.neuron import Neuron
from base.som_interface import SomInterface
from base import distance

ITERATION_MAX = 1000

class RecursiveSom(SomInterface):
    def __init__(self, map_dimensions, data_dimension, kernel_func, nb_clusters=0):
        self._data_dimension = data_dimension
        self._kernel_func = kernel_func
        self._nb_clusters = nb_clusters

        self._map_dimensions = []
        self._map_dimensions[:] = map_dimensions[:]

        self._neurons = []
        self._clusters = []
        self._dataset = None

        for position in som.generate_positions(map_dimensions):
            # Initialize _neurons positions in map
            neuron = Neuron(position=position)
            # Initialize _neurons coordinates
            coords = []
            for i in xrange(data_dimension):
                coords.append(random.random() * 10)
            neuron.set_data(coords)
            self._neurons.append(neuron)

        self._context = []
        for i in xrange(len(self._neurons)):
            self._context.append(0)

        for n in self._neurons:
            n.set_context(self._context)

    def __str__(self):
        res = "RecSOM"
        res += ", d: " + str(self._map_dimensions)
        if self._nb_clusters > 0:
            res += ", c: " + str(self._nb_clusters)

        return res

    def learn(self):
        data = self._dataset.data()
        dist = distance.euclidian

        sigma_init = dist(self._neurons[len(self._neurons) / 2].get_position(), \
                         self._neurons[len(self._neurons) - 1].get_position())
        t1 = 1000 / math.log(sigma_init)
        nu0 = 0.1
        t2 = 1000

        nu = lambda n: nu0 * math.exp(-n / t2)

        alpha = 2
        beta = 0.06

        activations = self._context

        iteration = 1
        while iteration <= ITERATION_MAX:
            x = data[int(random.random() * len(data))]

            sigma = sigma_init * math.exp(-iteration / t1)

            # Get the neuron with the highest activity for input vector x.
            best_activity = 0
            new_activations = []
            for n in self._neurons:
                di = alpha * dist(x, n.get_data()) \
                     + beta * dist(activations, n.get_context())
                activity = math.exp(- di)
                new_activations.append(activity)
                if activity > best_activity:
                    best_neuron = n
                    best_activity = activity

            # Update the neurons weights and contexts.
            for n in self._neurons:
                d = dist(n.get_position(), best_neuron.get_position(), \
                         dimension=len(n.get_position()))
                h = som.gaussian_kernel(d, sigma)

                ndata = n.get_data()
                for i in xrange(len(ndata)):
                    step = nu(iteration) * h * (x[i] - ndata[i])
                    ndata[i] += step

                ncontext = n.get_context()
                for i in xrange(len(ncontext)):
                    step = nu(iteration) * h * (activations[i] - ncontext[i])
                    ncontext[i] += step

            activations = new_activations

            iteration += 1

        som.clusterize(self._neurons, self._data_dimension, self._clusters, self._dataset, self._nb_clusters)

    def class_probability(self, point, c):
        """ Computes the probability that the nearest cluster of a given point
        to have a given label. The probability is given as 0 < p < 1. This
        probability is independant of any treshold. """

        # We first get the closest cluster.
        nearest_cluster = som.get_nearest_cluster(self._clusters, point)

        # We store every occurence of a label in a dictionary.
        # TODO: this dictionary should be in the cluster.
        frequences = {0: 0.0, 1: 0.0}
        for label in nearest_cluster._labels:
            frequences[label] += 1.0

        sum = frequences[0] + frequences[1]
        if sum == 0:
            return 0
        else:
            return frequences[c] / (frequences[0] + frequences[1])

    def classify (self, point):
        """ Computes the guessed label of a given point. """

        # TODO: We should have the possibility of defining a treshold for that.
        if self.class_probability(point, 1) > 0.5:
            return 1
        else:
            return 0
def main():
    rsom = RecursiveSom([5, 5], 1, None, 0)

if __name__ == "__main__":
    main()
