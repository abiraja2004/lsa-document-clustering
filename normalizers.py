import numpy as np
from math import log


class Normalizer(object):
  """
  Default normalizer (no normalisation)
  """

  @staticmethod
  def normalize(matrix):
    return matrix


class TFIDF(Normalizer):
  @staticmethod
  def normalize(matrix):
    """
    Normalizes the frequency counts as per the TF-IDF scheme (Salton and McGill, 1986)
    (term frequency–inverse document frequency)

    :param matrix: The source/term frequency matrix
    :type  matrix: numpy.matrix
    :return: The normalized source/term frequency matrix
    """
    # create a copy of it
    tfidf_matrix = matrix.copy()
    rows, cols = matrix.shape
    # gives the words for each document
    words_per_doc = np.sum(matrix, axis=0)
    # gives how many documents the word appears in
    docs_per_word = np.sum(np.asarray(matrix > 0, 'i'), axis=1)

    for i in range(rows):
      for j in range(cols):
        # calculate tfidf
        tfidf_matrix[i][j] = (matrix[i][j] / words_per_doc[j]) * log(float(cols) / docs_per_word[i])

    return tfidf_matrix


