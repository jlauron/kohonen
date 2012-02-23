import pylab
import numpy
from numpy import array
from numpy import matrix

from machinelearning.lib import dataset
from machinelearning.naivebayes import algorithm as nbc
from machinelearning.quadradicdiscriminant import algorithm as qdc
from machinelearning.knn import algorithm as knn

def f_compare(a, b):
    if a[1] > b[1]:
        return -1
    elif a[1] == b[1]:
        return 0
    else:
        return 1

class DecisionFunction:
    def draw_classifier(self, classifier):
        x = pylab.arange(-10.0, 10.0, 1.0)
        y = pylab.arange(-10.0, 10.0, 1.0)
        X, Y = pylab.meshgrid(x, y)

        z = numpy.zeros((20, 20))
        for i in xrange(-10, 10):
            for j in xrange(-10, 10):
                p = matrix((i, j)).T
                #p = array((i, j))
                z[i + 10, j + 10] = classifier.class_probability(p, "in")

        pylab.imshow(z, interpolation = "bilinear")
        pylab.axis("off")
        pylab.show()

class RocCurve:
    def __init__(self):
        pylab.plot([0, 1], [0, 1], 'k-')

    def draw_classifier(self, classifier, dataset, curve):
        L = []
        P = 0.0
        N = 0.0
        nb_real_positive = 0.0
        for i in range(0, len(dataset.data())):
            #p = matrix(dataset.data()[i]).T
            p = dataset.data()[i]
            res = classifier.classify(p)
            L.append((dataset.labels()[i], classifier.class_probability(p, "in")))
            #L.append((dataset.labels()[i], classifier.class_probability(dataset.data()[i], "in")))
            if res == dataset.labels()[i]:
                nb_real_positive += 1
            if dataset.labels()[i] == "in":
                P += 1
            else:
                N += 1

        L.sort(f_compare)
        FP = 0.0
        TP = 0.0
        Rx = []
        Ry = []
        f_prev = -1
        for i in range(0, len(L)):
            if f_prev == -1 or L[i][1] != f_prev:
                Rx.append(FP / N)
                Ry.append(TP / P)
                f_prev = L[i][1]
            if L[i][0] == "in":
                TP += 1
            else:
                FP += 1
        Rx.append(FP / N)
        Ry.append(TP / P)

        pylab.plot(Rx, Ry, curve)

    def show(self):
        pylab.show()

def draw_function():
    df = DecisionFunction()
    #classifier = nbc.NaiveBayes()
    classifier = qdc.QuadraticDiscriminantClassifier(2)
    #classifier = knn.KNN(3)
    datas = dataset.load ("../datasets/gaussian10k.data")
    #classifier.load(datas[1000:])
    classifier.load(datas[9000:])
    classifier.learn()
    df.draw_classifier(classifier)

def draw_all_roc():
    roc = RocCurve()

    datas = dataset.load ("../datasets/donut1k-20.data")
    test_set = datas[0:900]

#    classifier = qdc.QuadraticDiscriminantClassifier(2)
#    classifier.load(datas[4500:])
#    classifier.learn()
#    roc.draw_classifier(classifier, test_set, "r-")

    classifier = nbc.NaiveBayes()
    classifier.load(datas[900:])
    classifier.learn()
    roc.draw_classifier(classifier, test_set, "r-")

#    test_set = datas[0:900]
#
#    classifier = knn.KNN(3)
#    classifier.load(datas[900:])
#    classifier.learn()
#    roc.draw_classifier(classifier, test_set, "r-")

    roc.show()

#draw_function()
#draw_all_roc()
