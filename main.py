import os.path
import filters
from matrices import *
from algorithms import LSA
from normalizers import *
from visualize import Visualize

import nltk


class Pipeline:
  """
  Class which assembles all the parts together to build a document clustering
  system.
  """

  def __init__(self):
    # set up some default filters
    self.filterz = [filters.StopWordFilter("stopwords2.data"), filters.BasicWordFilter()]

  def cluster_documents(self, docs, dimensions_reduction, clusters, normalizer=TFIDF):
    # calculate the frequency matrix
    term_matrix = FrequencyMatrix(docs, filters=self.filterz)
    # Set-up Latent Semantic Analysis class
    lsa = LSA(term_matrix, normalizer=TFIDF)
    # Decompose term matrix into SVD
    svd = lsa.decompose()
    # Reduce the dimensions of the SVD
    rsvd = lsa.reduce_svd(dimensions_reduction, svd)
    # Cluster the data
    centroids, doc_clusters, labels = lsa.cluster(clusters, rsvd, dim=dimensions_reduction)
    # Visualize it
    vis = Visualize()
    vis.plot_documents(rsvd, labels, doc_clusters, len(centroids))
    vis.show()

  def summarize_sentences(self, sentences, dimensions_reduction, no_of_sentences=3, normalizer=TFIDF):
    # calculate the frequency matrix
    term_matrix = FrequencyMatrix(sentences, filters=self.filterz)
    # Set-up Latent Semantic Analysis class
    lsa = LSA(term_matrix, normalizer=TFIDF)
    # Decompose matrix into SVD
    svd = lsa.decompose()
    # Calculate the most significant sentences based on SVD
    # more details see algorithms.py
    scores = lsa.significant_terms(dimensions_reduction, svd)
    best = sorted(scores, reverse=True)
    for i in range(0, no_of_sentences):
      print(best[i][1], end=' ')


def create_sentence_sources(path, tokenizer=None):
  """Loads up a document and splits it into sentences"""
  document = open(path, encoding='utf-8').read()
  sources = []
  sentences = nltk.tokenize.sent_tokenize(document)
  i = 0
  for s in sentences:
    sources.append(SentenceSource(s, "Sentence " + str(i), tokenizer=tokenizer))
    i += 1
  return sources


def create_document_sources(directory, tokenizer=None):
  """Loads up the documents for clustering based on similarity"""
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
sum_file = "obama.txt"
clusters = 4
svd_dim = 2

# tokenizer which breaks words down
tokenizer = langprocess.NLTKTokenizer
# normalizer of frequency matrix
normalizer = TFIDF

main = Pipeline()

print("------------------------------")
print("Text summarization example\n")
print("Enter file to summarize\n (Press enter for default: " + sum_file + ")")
sum_file = input_or_default(sum_file)

sentences = create_sentence_sources(os.path.join(dir, sum_file), tokenizer=langprocess.NLTKTokenizer)

print("Summary of", sum_file)
main.summarize_sentences(sentences, 10)
print("")

print("------------------------------------")
print("Document clustering example\n")
print("Enter directory to load examples from\n(Press enter for default: " + dir + ")")

dir = input_or_default(dir)

docs = create_document_sources(dir, tokenizer=langprocess.NLTKTokenizer)
print("Enter the number of clusters to detect: \n(Press enter for default: " + str(clusters) + ")")
clusters = int(input_or_default(clusters))

print("Enter SVD dimensions for clustering: \n(Press enter for default: " + str(svd_dim) + ")")
svd_dim = int(input_or_default(svd_dim))

main.cluster_documents(docs, svd_dim, clusters)
