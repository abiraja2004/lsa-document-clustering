import numpy as np
from functools import reduce
import langprocess


class Source(object):

  """Represents a word source eg. a document or a sentence"""

  def __init__(self, source, name, tokenizer=None):
    self.text = source
    self.name = name
    self.words = None
    self.no_words = 0
    if tokenizer is None:
      self.tokenizer = langprocess.RegexTokenizer
    else:
      self.tokenizer = tokenizer

  def get_words(self, filters=None):
    """
    Gets all the words within the source

    :param filters: Collection of word filters which tell
                    which words can make it into
                    the list and which cannot
    :type filters: list[WordFilter]
    :return: List of words
    """
    # find all words in our source
    word_list = self.tokenizer.get_word_tokens(self.text)

    # filter our words so that only those who pass
    # the filter are accepted
    if filters is not None:
      for i, word in enumerate(word_list):
        for _filter in filters:
          if not _filter.accept(word):
            del word_list[i]
            break

    self.words = word_list
    self.no_words = len(word_list)

    return word_list

  def get_name(self):
    """Get name of the source"""
    return self.name

  def frequency_count(self, filters=None, words=None):
    """
    Computes a frequency-count dictionary of words vs number of times
    they appear within this source

    :param filters: Collection of filters (conditions to remove words)
    :type  filters: list[WordFilter]
    :param   words: Words to calculate frequency of
    :type    words: list[str]
    """
    if words is None:
      words = self.get_words(filters=filters)

    # calculate the frequency list for them
    frequencies = [words.count(i) for i in words]
    # convert to a dictionary
    freqs = dict(zip(words, frequencies))

    return freqs


class DocSource(Source):

  """ Represents a document source """
  def __init__(self, path, name, tokenizer=None):
    self.file = open(path, encoding='utf-8').read()
    super(DocSource, self).__init__(self.file, name, tokenizer)


class SentenceSource(Source):

  """Represent a sentence source"""
  def __init__(self, contents, name, tokenizer=None):
    super(SentenceSource, self).__init__(contents, name, tokenizer)


class FrequencyMatrix:

  def __init__(self, sources, filters=None):
    """
    Builds a frequency matrix for words and sources. Sources can be sentences or
    documents.

    The matrix is |unique words| x |sources|

    :param sources: The text source where to build the frequency matrix
    :type  sources: list[Source]
    :param filters: Collection of filters (conditions to remove words)
    :type  filters: list[WordFilter]
    """
    # get all words from all documents
    all_words = reduce(list.__add__, [s.get_words(filters=filters) for s in sources])
    # get frequency counts for each source
    freq_counts = [s.frequency_count(filters=filters) for s in sources]
    # de-duplicate
    unique_words = sorted(set(all_words))
    no_sources = len(sources)
    no_words = len(unique_words)

    # create source-term matrix
    matrix = np.zeros((no_words, no_sources))
    # stores which column each word should be stored in
    word_columns = {word: i for i, word in enumerate(unique_words)}

    # go through all the sources and build a matrix of occurrences of each word
    # in each of them
    for i, freq_count in enumerate(freq_counts):
      for word, count in freq_count.items():
        matrix[word_columns[word]][i] = count

    self.topics = [s.get_name() for s in sources]
    self.matrix = matrix
    self.words = unique_words
    self.sources = sources

  def get_topics(self):
    """
    Returns the topics associated with the source

    In most of our examples topics will be filenames
    :return: The topics
    """
    return self.topics

  def get_sources(self):
    """
    Return the sources in this frequency matrix
    :return: The sources
    """
    return self.sources

  def to_array(self):
    """
    Converts this to an array
    :return: Array
    """
    return np.array(self.matrix)

  def wordlist(self):
    """
    Returns all the words in the sources
    :return: List of all the words in the matrix
    """
    return self.words
