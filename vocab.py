# Run with (e.g.)
# $ python3 vocab.py --data_name fashionIQ

import os
import argparse
import pickle
from collections import Counter
import json
import nltk
from config import MAIN_DIR, FASHIONIQ_ANNOTATION_DIR, SHOES_ANNOTATION_DIR, CIRR_ANNOTATION_DIR, FASHION200K_ANNOTATION_DIR, cleanCaption


################################################################################
# *** LOCATION OF DATASET ANNOTATIONS
################################################################################
# List of files containing the text/captions from which to build the vocabs, for
# each dataset.

ANNOTATIONS = {
  'fashionIQ': [f'{FASHIONIQ_ANNOTATION_DIR}/captions/cap.{fc}.train.json' for fc in ['dress','shirt','toptee']],
  'shoes': [f'{SHOES_ANNOTATION_DIR}/triplet.train.json'],
  'cirr': [f'{CIRR_ANNOTATION_DIR}/captions/cap.rc2.train.json'],
  'fashion200K': [f'{FASHION200K_ANNOTATION_DIR}/{fc}_train_detect_all.txt' for fc in ['dress', 'skirt', 'jacket', 'pants', 'top']]
}


################################################################################
# *** VOCABULARY CLASS
################################################################################

class Vocabulary(object):
  """Simple vocabulary wrapper."""

  def __init__(self):
    self.idx = 0
    self.word2idx = {}
    self.idx2word = {}

  def add_word(self, word):
    if word not in self.word2idx:
      self.word2idx[word] = self.idx
      self.idx2word[self.idx] = word
      self.idx += 1

  def __call__(self, word):
    return self.word2idx.get(word, self.word2idx['<unk>'])

  def __len__(self):
    return len(self.word2idx)


################################################################################
# *** FUNCTIONS TO COLLECT THE CAPTIONS (depends on each dataset)
################################################################################

def from_fashionIQ_json(p):
  with open(p, "r") as jsonfile:
    ann = json.loads(jsonfile.read())
  captions = [cleanCaption(a["captions"][0]) for a in ann] # caption 1
  captions += [cleanCaption(a["captions"][1]) for a in ann] # caption 2
  return captions


def from_shoes_json(p):
  with open(p, "r") as jsonfile:
    ann = json.loads(jsonfile.read())
  captions = [cleanCaption(a["RelativeCaption"]) for a in ann]
  return captions


def from_cirr_json(p):
  with open(p, "r") as jsonfile:
    ann = json.loads(jsonfile.read())
  captions = [cleanCaption(a["caption"]) for a in ann]
  return captions


def from_fashion200K_txt(p):
    with open(p, 'r') as file:
        content = file.read().splitlines()
    # first line is the image filename, second line is the detection score
    caption = [cleanCaption(line.split('\t')[-1]) for line in content]
    return caption


def from_txt(txt):
  captions = []
  with open(txt, 'rb') as f:
    for line in f:
      captions.append(line.strip())
  return captions


################################################################################
# *** BUILD VOCABULARY
################################################################################

def build_vocab(data_name, threshold=0):
  """
  Build vocabulary from annotation files.

  Input:
    - data_name: name of the dataset for which to build the vocab.
    - threshold: minimal number of occurrences for a word to be included in the
      vocab.

  Output:
    vocab: Vocabulary object
  """

  # Initialization
  counter = Counter()

  # Gather all the texts (captions) on which the vocab will be based
  for p in ANNOTATIONS[data_name]:
    if data_name == 'fashionIQ':
      captions = from_fashionIQ_json(p)
    elif data_name == 'shoes':
      captions = from_shoes_json(p)
    elif data_name == 'fashion200K':
      captions = from_fashion200K_txt(p)
    elif data_name == 'cirr':
      captions = from_cirr_json(p)
    else:
      captions = from_txt(p)

    # Process the captions: tokenize & register words
    for caption in captions:
      tokens = nltk.tokenize.word_tokenize(caption.lower())
      counter.update(tokens)

  # Discard words for which the number of occurrences is smaller than a provided
  # threshold.
  words = [word for word, cnt in counter.items() if cnt >= threshold]
  print('Vocabulary size: {}'.format(len(words)))

  # Create a vocab wrapper and add some special tokens.
  vocab = Vocabulary()
  vocab.add_word('<pad>')
  vocab.add_word('<start>')
  vocab.add_word('<and>') # to link several captions together
  vocab.add_word('<end>')
  vocab.add_word('<unk>')
  # fashion200K specific case: add "replace" and "with" to vocab
  if data_name == 'fashion200K':
      vocab.add_word('replace')
      vocab.add_word('with')

  # Add words to the vocabulary.
  for word in words:
    vocab.add_word(word)

  return vocab


def main(data_name, threshold, vocab_dir):
  # create the required vocab
  vocab = build_vocab(data_name, threshold=threshold)
  # create the directory in which the vocab should be saved
  if not os.path.isdir(vocab_dir):
    os.makedirs(vocab_dir)
  # create the vocab file
  vocab_path = os.path.join(vocab_dir, f'{data_name}_vocab.pkl')
  with open(vocab_path, 'wb') as f:
    pickle.dump(vocab, f, pickle.HIGHEST_PROTOCOL)
  print("Saved vocabulary file to ", vocab_path)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument('--data_name', default='fashionIQ', choices=('fashionIQ', 'shoes', 'cirr', 'fashion200K'), help='Name of the dataset for which to build the vocab (fashionIQ|shoes|cirr|fashion200K)')
  parser.add_argument('--vocab_dir', default=MAIN_DIR + '/vocab/', help='Root directory for the vocab files.')
  parser.add_argument('--threshold', default=0, type=int, help="Minimal number of occurrences for a word to be included in the vocab.")
  opt = parser.parse_args()
  main(opt.data_name, opt.threshold, opt.vocab_dir)
