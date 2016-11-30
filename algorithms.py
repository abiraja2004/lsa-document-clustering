import scipy
import scipy.cluster
import numpy as np
import math


class LSA:
  def __init__(self, matrix, normalizer=None):
    """
    Initialise the LSA pipeline
    :type matrix: matrices.FrequencyMatrix
    :param normalizer: Any algorithm to normalize the frequency matrix
    :type normalizer: Normalizer
    """
    self.freqmatrix = matrix
    if normalizer:
      self.matrix = normalizer.normalize(matrix.to_array())
    else:
      self.matrix = matrix.to_array()

  def decompose(self):
    """
    Computes the Singular Value Decomposition matrix

    :return: The SVD matrix
    """
    return scipy.linalg.svd(self.matrix)

  def reduce_svd(self, k, svd=None):
    """
    Performs dimensionality reduction to the SVD matrix.

    :param svd: The Singular Value Decomposition Matrix
    :param k: Dimensions to keep
    :type k: int

    :return: The dimensionality reduced SVD
    """

    if svd:
      u, s, vt = svd
    else:
      u, s, vt = self.decompose()

    reduced = vt[0:k + 1, :]

    return u, reduced

  def significant_terms(self, k, svd):

    if svd:
      u, s, vt = svd
    else:
      u, s, vt = self.decompose()

    # compute the reduced sigma squared vector
    sigma_reduced = [s ** 2 for i, s in enumerate(s[: k + 1])]

    scores = []
    topics = self.freqmatrix.get_topics()

    for i, col_vector in enumerate(vt.T):
      # compute the column vector from svd times the squared
      # singular values
      s_k = sum(s * v ** 2 for s, v in zip(sigma_reduced, col_vector))
      score = math.sqrt(s_k)
      scores.append((score, topics[i]))

    return scores

  def cluster(self, no_clusters, rsvd, dim=2):
    """
    Clusters the documents using the SVD

    :param no_clusters: Number of documents to cluster into
    :param rsvd: The reduced SVD matrix
    :param dim: The dimensions to use when clustering
    :return: A tuple (centroids, doc_clusters, labels)
    """
    u, vt = rsvd

    # prepare the data for kmeans clustering
    data_kmeans = np.dstack((vt[i] for i in range(1, dim + 1)))[0]

    # run kmeans+ on the data
    centroids, doc_clusters = scipy.cluster.vq.kmeans2(data_kmeans, no_clusters, minit="points")

    # doc_clusters is an array indicating to which cluster each document belongs
    return centroids, doc_clusters, [label for label in self.freqmatrix.get_topics()]
