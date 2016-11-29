import os.path
import filters
from matrices import *
from algorithms import *
from normalizers import *
import matplotlib.cm
import matplotlib.colors
import matplotlib.pyplot as plt


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

  def show(self):
    plt.axis((-1, 1, 1, -1,))
    plt.show()


class Pipeline:

  """
  Class which assembles all the parts together to build a document clustering
  system.
  """

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
    vis.show()

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
