import os
import torch
import helpers
import itertools

# Default word tokens
PAD_token = 0  # Used for padding short sentences
SOS_token = 1  # Start-of-sentence token
EOS_token = 2  # End-of-sentence token


class Voc:
  def __init__(self, name):
    self.name = name
    self.trimmed = False
    self.word2index = {}
    self.word2count = {}
    self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
    self.num_words = 3  # Count SOS, EOS, PAD

  def addSentence(self, sentence):
    for word in sentence.split(' '):
      self.addWord(word)

  def addWord(self, word):
    if word not in self.word2index:
      self.word2index[word] = self.num_words
      self.word2count[word] = 1
      self.index2word[self.num_words] = word
      self.num_words += 1
    else:
      self.word2count[word] += 1

  # Remove words below a certain count threshold
  def trim(self, min_count):
    if self.trimmed:
      return
    self.trimmed = True

    keep_words = []

    for k, v in self.word2count.items():
      if v >= min_count:
        keep_words.append(k)

    print('keep_words {} / {} = {:.4f}'.format(
        len(keep_words), len(self.word2index),
        len(keep_words) / len(self.word2index)))

    # Reinitialize dictionaries
    self.word2index = {}
    self.word2count = {}
    self.index2word = {PAD_token: "PAD", SOS_token: "SOS", EOS_token: "EOS"}
    self.num_words = 3  # Count default tokens

    for word in keep_words:
      self.addWord(word)


# Read query/response pairs and return a voc object
def readVocs(datafile, corpus_name):
  print("Reading lines...")
  # Read the file and split into lines
  lines = open(datafile, encoding='utf-8').\
      read().strip().split('\n')
  # Split every line into pairs and normalize
  pairs = [[helpers.normalizeString(s) for s in l.split('\t')] for l in lines]
  voc = Voc(corpus_name)
  return voc, pairs


def load_chinese_dialog_pairs(corpus_name):
  voc = Voc(corpus_name)
  import json
  path = os.path.join("data", "lccc", "LCCD_filtered.json")
  file = open(path, 'r', encoding="utf8", errors='ignore')
  pairs = json.load(file)
  pairs = pairs[:100000]
  for pair in pairs:
    voc.addSentence(pair[0])
    voc.addSentence(pair[1])
  return voc, pairs


# Using the functions defined above, return a populated voc object and pairs list
def loadPrepareData(corpus, corpus_name, datafile):
  print("Start preparing training data ...")
  voc, pairs = readVocs(datafile, corpus_name)
  print("Read {!s} sentence pairs".format(len(pairs)))
  pairs = helpers.filterPairs(pairs)
  print("Trimmed to {!s} sentence pairs".format(len(pairs)))
  print("Counting words...")
  for pair in pairs:
    voc.addSentence(pair[0])
    voc.addSentence(pair[1])
  print("Counted words:", voc.num_words)
  return voc, pairs


# Wrapper for load English dialogs
def load_english_dialog_pairs(corpus_name):
  # corpus_name = "cornell movie-dialogs corpus"
  corpus = os.path.join("data", corpus_name)
  datafile = os.path.join(corpus, "formatted_movie_lines.txt")
  return loadPrepareData(corpus, corpus_name, datafile)


def indexesFromSentence(voc, sentence):
  return [voc.word2index[word] for word in sentence.split(' ')] + [EOS_token]


def zeroPadding(l, fillvalue=PAD_token):
  return list(itertools.zip_longest(*l, fillvalue=fillvalue))


def binaryMatrix(l, value=PAD_token):
  m = []
  for i, seq in enumerate(l):
    m.append([])
    for token in seq:
      if token == PAD_token:
        m[i].append(0)
      else:
        m[i].append(1)
  return m


# Returns padded input sequence tensor and lengths
def inputVar(l, voc):
  indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
  lengths = torch.tensor([len(indexes) for indexes in indexes_batch])
  padList = zeroPadding(indexes_batch)
  padVar = torch.LongTensor(padList)
  return padVar, lengths


# Returns padded target sequence tensor, padding mask, and max target length
def outputVar(l, voc):
  indexes_batch = [indexesFromSentence(voc, sentence) for sentence in l]
  max_target_len = max([len(indexes) for indexes in indexes_batch])
  padList = zeroPadding(indexes_batch)
  mask = binaryMatrix(padList)
  mask = torch.BoolTensor(mask)
  padVar = torch.LongTensor(padList)
  return padVar, mask, max_target_len


# Returns all items for a given batch of pairs
def batch2TrainData(voc, pair_batch):
  pair_batch.sort(key=lambda x: len(x[0].split(" ")), reverse=True)
  input_batch, output_batch = [], []
  for pair in pair_batch:
    input_batch.append(pair[0])
    output_batch.append(pair[1])
  inp, lengths = inputVar(input_batch, voc)
  output, mask, max_target_len = outputVar(output_batch, voc)
  return inp, lengths, output, mask, max_target_len
