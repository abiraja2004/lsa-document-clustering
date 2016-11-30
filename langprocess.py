import nltk.stem.wordnet as wordnet
import nltk
import os
import re

# set the nltk data path to local dir
nltk.data.path= [os.path.dirname(os.path.abspath(__file__)) + os.sep + "nltk_data"]


class Tokenizer(object):

  @staticmethod
  def get_word_tokens(text):
    return None


class RegexTokenizer(Tokenizer):

  """Simple regex based tokenizer"""
  @staticmethod
  def get_word_tokens(text):
    # find all words in our source
    word_list = re.compile('\w+').findall(text)
    # convert them to lowercase
    word_list = list(map(str.lower, word_list))

    return list(set(word_list))


class NLTKTokenizer(Tokenizer):

  lemmatizer = wordnet.WordNetLemmatizer()

  @staticmethod
  def get_word_tokens(text):
    tokens = nltk.word_tokenize(text)
    lemmatized = list(map(NLTKTokenizer.lemmatizer.lemmatize,tokens))
    return list(set(lemmatized))