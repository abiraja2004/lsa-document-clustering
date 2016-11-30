import nltk
import re


class WordFilter(object):
  def accept(self, word): raise NotImplementedError


class StopWordFilter(WordFilter):

  """Filters common English stopwords"""
  def __init__(self, stopwords_file):
    words = []
    # open the file and load the stopwords
    with open(stopwords_file, encoding='utf-8') as f:
      for line in f:
        words.append(line.lstrip().rstrip())
    self.stopwords = set(words)

  def accept(self, word):
    return word not in self.stopwords


class NLTKStopWordFilter(WordFilter):

  def __init__(self):
    self.stopwords = set(nltk.corpus.stopwords.words('english'))

  def accept(self, word):
    return word not in self.stopwords


class NumberFilter(WordFilter):

  """Removes numbers"""

  def accept(self, word):
    return not word.isdigit()


class BasicWordFilter(WordFilter):

  """Basic simple filter"""

  def accept(self, word):
    return len(word) > 1 and re.match("\w", word)
