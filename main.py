import os.path
import filters
from matrices import *
from algorithms import *
from normalizers import *
import matplotlib.cm
import matplotlib.colors
import matplotlib.pyplot as plt
import nltk

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
    self.filterz = [filters.StopWordFilter("stopwords2.data"), filters.NLTKStopWordFilter(), filters.BasicWordFilter()]
    # calculate the frequency matrix
    self.term_matrix = FrequencyMatrix(docs, filters=self.filterz)

  def run(self, dimensions_reduction, clusters, normalizer=TFIDF):
    # Set-up Latent Semantic Analysis class
    lsa = LSA(self.term_matrix, normalizer=TFIDF)
    # Decompose term matrix into SVD
    svd = lsa.decompose()
    # Reduce the dimensions of the SVD
    rsvd = lsa.reduce_svd(dimensions_reduction, svd)
    ll = lsa.calculate_ranks(dimensions_reduction, svd)
    best = sorted(ll)
    print(ll)
    # Cluster the data
    centroids, doc_clusters, labels = lsa.cluster(clusters, rsvd, dim=dimensions_reduction)
    vis = Visualize()
    vis.plot_documents(rsvd, labels, doc_clusters, len(centroids))
    vis.show()


def create_sentence_sources(path, tokenizer=None):
  document = open(path).read()
  sources = []
  sentences = nltk.tokenize.sent_tokenize(document)
  i = 0
  for s in sentences:
    sources.append(SentenceSource(s, "Sentence " + str(i), tokenizer=tokenizer))
    i += 1
  return sources


def create_document_sources(directory, tokenizer=None):
  docs = []
  for filename in os.listdir(directory):
    if filename.endswith(".txt"):
      docs.append(DocSource(os.path.join(directory, filename), filename, tokenizer=tokenizer))
  return docs


def input_or_default(default):
  inp = input("")
  if inp.rstrip().lstrip() != "":
    return inp
  else:
    return default

######################
# Parameters section #
######################

dir = "presidents_rivers"
clusters = 3
svd_dim = 2


sentences = create_sentence_sources("presidents_rivers/bush.txt", tokenizer=langprocess.NLTKTokenizer)

main = Pipeline(sentences)
main.run(svd_dim, clusters)
pass

# tokenizer which breaks words down
tokenizer = langprocess.NLTKTokenizer
# normalizer of frequency matrix
normalizer=TFIDF

print("Enter directory to load examples from\n(Press enter for default: " + dir + ")")

dir = input_or_default(dir)

docs=create_document_sources(dir, tokenizer=langprocess.NLTKTokenizer)
print("Enter the number of clusters to detect: \n(Press enter for default: " + str(clusters) + ")")
clusters = int(input_or_default(clusters))

print("Enter SVD dimensions for clustering: \n(Press enter for default: " + str(svd_dim) + ")")
svd_dim = int(input_or_default(svd_dim))

main = Pipeline(docs)
main.run(svd_dim, clusters)
