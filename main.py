import scipy
import scipy.cluster
import os.path
import filters
from matrices import *
from math import log
import matplotlib.cm
import matplotlib.colors
import matplotlib.pyplot as plt


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


class Visualize:
  def __init__(self):
    self.colours = ['r', 'b', 'g']
    self.cl = 0

  def plot(self, xs, ys, labels, colours=None):
    plt.scatter(xs, ys, c=colours)
    if labels is not None:
      for label, x, y in zip(labels, xs, ys):
        plt.annotate(
          label,
          xy=(x, y), xytext=(-30, 30),
          textcoords='offset points', ha='right', va='bottom',
          arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))
    self.cl += 1

  def plot_cluster(self, centroids):
    self.plot(centroids[:, 0], centroids[:, 1], labels=None, colours=['g'] * centroids.shape[1])

  def plot_documents(self, svd, names, doc_clusters, no_clusters):
    u, vt = svd
    pts = vt
    colormap = plt.get_cmap("hsv")
    norm = matplotlib.colors.Normalize(vmin=0, vmax=no_clusters)
    scalarMap = matplotlib.cm.ScalarMappable(cmap=colormap, norm=norm)
    self.plot(pts[1], pts[2], names, colours=[scalarMap.to_rgba(i) for i in doc_clusters])

  @staticmethod
  def show():
    plt.axis((-1, 1, 1, -1,))
    plt.show()


class Pipeline:

  def __init__(self, docs):
    # set up some default filters
    self.filterz = [filters.StopWordFilter("stopwords2.data"), filters.BasicWordFilter()]
    # calculate the frequency matrix
    self.term_matrix = FrequencyMatrix(docs, filters=self.filterz)

  def run(self, dimensions_reduction, clusters):
    # Set-up Latent Semantic Analysis class
    lsa = LSA(self.term_matrix, normalizer=TFIDF)
    # Decompose term matrix into SVD
    svd = lsa.decompose()
    # Reduce the dimensions of the SVD
    rsvd = lsa.reduce_svd(dimensions_reduction, svd)
    # Cluster the data
    centroids, doc_clusters, labels = lsa.cluster(clusters, rsvd, dim=dimensions_reduction)
    vis = Visualize()
    vis.plot_documents(rsvd, labels, doc_clusters, len(centroids))
    Visualize.show()


def load_documents(dir):
  global docs
  docs = []
  for filename in os.listdir(dir):
    if filename.endswith(".txt"):
      docs.append(DocSource(os.path.join(dir, filename), filename))


def input_or_default(default):
  inp = input("")
  if inp.rstrip().lstrip() != "":
    return inp
  else:
    return default


dir = "presidents_rivers"
clusters = 3
svd_dim = 2

print("Enter directory to load examples from\n(Press enter for default: " + dir + ")")

dir = input_or_default(dir)

load_documents(dir)
print("Enter the number of clusters to detect: \n(Press enter for default: " + str(clusters) + ")")
clusters = int(input_or_default(clusters))

load_documents(dir)
print("Enter SVD dimensions for clustering: \n(Press enter for default: " + str(svd_dim) + ")")
svd_dim = int(input_or_default(svd_dim))

main = Pipeline(docs)
main.run(svd_dim, clusters)
