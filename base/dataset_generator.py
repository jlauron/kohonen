import random
import pylab
import scipy
import sys

from base import dataset

def generate_gaussian (nb_points, dimension, mu, sigma, center=(0, 0)):
    """ Generate Gaussian array of nb_points in dimension where mu is the mean
    and sigma the standard deviation """
    points = scipy.zeros((nb_points, dimension))
    for i in xrange(0, nb_points):
        for j in xrange(0, dimension):
            points[i, j] = random.gauss(mu, sigma) + center[j]

    return points

def export(class_name, points):
    for i in range(0, len(points)):
        sys.stdout.write(class_name)
        for j in xrange(0, points.ndim):
            sys.stdout.write("," + str(points[i, j]))
        sys.stdout.write("\n")

def main():
    points = generate_gaussian(1000, 2, 0, 2, center=(10, 0))
    pylab.plot (points[:,0], points[:,1], 'r+')
    #export("Classe A", points)
    points2 = generate_gaussian(1000, 2, 0, 2, center=(5, 5))
    pylab.plot (points2[:,0], points2[:,1], 'b+')
    #export("Classe C", points)
    points3 = generate_gaussian(1000, 2, 0, 2, center=(0, 10))
    pylab.plot (points3[:,0], points3[:,1], 'y+')
    points4 = generate_gaussian(1000, 2, 0, 2, center=(0, 0))
    pylab.plot (points4[:,0], points4[:,1], 'g+')
    pylab.axis([-10, 20, -10, 20])
    pylab.show()

    labels = []
    for i in xrange(len(points)):
        labels.append(0)
    for i in xrange(len(points2)):
        labels.append(1)
    for i in xrange(len(points3)):
        labels.append(2)
    for i in xrange(len(points4)):
        labels.append(3)

    points = scipy.concatenate ((points, points2))
    points = scipy.concatenate ((points, points3))
    points = scipy.concatenate ((points, points4))

    data = dataset.Dataset (points, labels)
    data.random ()

    dataset.save (data, "../datasets/4gaussians1k.data")

def main2gaussians ():
    #export("Classe A", points)
    points2 = generate_gaussian(1000, 2, 0, 2, center=(5, 5))
    pylab.plot (points2[:,0], points2[:,1], 'b+')
    #export("Classe C", points)
    points4 = generate_gaussian(1000, 2, 0, 2, center=(0, 0))
    pylab.plot (points4[:,0], points4[:,1], 'g+')
    pylab.axis([-10, 20, -10, 20])
    pylab.show()

    points = scipy.concatenate ((points2,points4))

    labels = []
    for i in xrange(len(points2)):
        labels.append(0)
    for i in xrange(len(points4)):
        labels.append(1)
    data = dataset.Dataset (points, labels)
    data.random ()

    dataset.save (data, "../datasets/2gaussians1k.data")

if __name__ == "__main__":
    main2gaussians()
