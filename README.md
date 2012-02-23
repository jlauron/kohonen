#Kohonen Self Organizing Maps (SOM) - python algorithms

This project contains a python implementation of several algorithms related
to the self organizing maps of kohonen. For more information on Kohonen maps
refer to (http://en.wikipedia.org/wiki/Self-organizing_map).

**base/**

  contains base python code for manipulating datasets, defining the algorithms
  common interface and other machine learning tools and helpers (like cross
  validation, distance calculations, etc).

**algorithms/**

  contains all algorithms implementation:
    - kohonen: kohonen som maps algorithm implementation
    variants:
    - recsom: recursive self organizing maps variant implementation
    other algorithms:
    - knn: k-nearest neighbors algorithm implementation
    - kmeans: k-means algorithm implementation
    - stochastickmeans: stochastic k-means variant of k-means algorithm implementaion
    - lvq: linear vector quantization algorithm implementation
    - svm: support vector machines algorithm interface

**datasets/**

  contains sample dataset to play with the algorithms and compare their results.

Note:
In order to get all python scripts to work, the root directory (kohonen/) must
be in the PYTHONPATH.
