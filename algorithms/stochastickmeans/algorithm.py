import math
import pylab
import copy
import random

from base.som_interface import SomInterface
from base.cluster import Cluster
from base import dataset

class StochasticKmeans(SomInterface):
    def __init__(self, k):
        self.__k = k
        self._clusters = []
        for i in xrange(self.__k):
            new_cluster = Cluster()
            self._clusters.append(new_cluster)

    def draw_all (self):
        self._dataset.draw()
        for cluster in self._clusters:
            cluster.draw_center()
        pylab.axis([-10, 20, -10, 20])
        pylab.show()

    def learn(self):
        data = self._dataset.data()
        dimension = len(data[0])

        clusters = self._clusters
        for cluster in clusters:
            i = int(random.random() * len(data))
            cluster.set_center(copy.deepcopy(data[i]))

        prev_centroids = copy.deepcopy(clusters)

        nb_iterations = 1
        stable_centers = False
        while (not stable_centers) and (nb_iterations <= 100000):
            i = int(random.random() * len(data))
            gradient_step = 1 / math.sqrt(nb_iterations)

            minimal_distance = clusters[0].distance_with(data[i])
            nearest_centroid = 0
            for j in xrange(1, len(clusters)):
                temp_distance = clusters[j].distance_with(data[i])
                if temp_distance < minimal_distance:
                    minimal_distance = temp_distance
                    nearest_centroid = j

            for j in xrange(dimension):
                clusters[nearest_centroid][j] -= 2 * gradient_step * (clusters[nearest_centroid][j] - data[i][j])

            if nb_iterations % 5 == 0:
                stable_centers = True
                for k in xrange(0, len(clusters)):
                    stable_centers = stable_centers and \
                        self.compute_distance(prev_centroids[k], clusters[k]) < 0.01
                prev_centroids = copy.deepcopy(clusters)

            nb_iterations += 1

        self._clusters = clusters

        return nb_iterations


def test ():
    data = dataset.load("../../datasets/4gaussians1k.data")

    classifier = StochasticKmeans(4)
    classifier.load(data)

    print "Stochastic Kmeans finished in %i iterations over the dataset" % \
        (classifier.learn() / len(data))

    classifier.draw_all()
    pylab.show()


if __name__ == "__main__":
    test()
