import os.path
import filters
from matrices import *
from algorithms import LSA
from normalizers import *

import nltk





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

  def cluster_documents(self, dimensions_reduction, clusters, normalizer=TFIDF):
    # Set-up Latent Semantic Analysis class
    lsa = LSA(self.term_matrix, normalizer=TFIDF)
    # Decompose term matrix into SVD
    svd = lsa.decompose()
    # Reduce the dimensions of the SVD
    rsvd = lsa.reduce_svd(dimensions_reduction, svd)
    ll = lsa.significant_terms(dimensions_reduction, svd)
    best = sorted(ll, reverse=True)
    print(best[0],best[1],best[2])
    # Cluster the data
    centroids, doc_clusters, labels = lsa.cluster(clusters, rsvd, dim=dimensions_reduction)
    vis = Visualize()
    vis.plot_documents(rsvd, labels, doc_clusters, len(centroids))
    vis.show()

  def summarize_sentences(self, dimensions_reduction, normalizer=TFIDF):
    # Set-up Latent Semantic Analysis class
    lsa = LSA(self.term_matrix, normalizer=TFIDF)
    # Decompose matrix into SVD
    svd = lsa.decompose()
    # Calculate the most significant sentences based on SVD
    # more details see algorithms.py

    pass


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
