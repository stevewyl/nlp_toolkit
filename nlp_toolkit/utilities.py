"""
some nlp process utilty functions
"""

import io
import time
import logging
import numpy as np
from itertools import groupby

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
logger = logging.getLogger('nlp_toolkit')


def shorten_word(word):
    """
    Shorten groupings of 3+ identical consecutive chars to 2, e.g. '!!!!' --> '!!'
    """

    # must have at least 3 char to be shortened
    if len(word) < 3:
        return word
    # find groups of 3+ consecutive letters
    letter_groups = [list(g) for k, g in groupby(word)]
    triple_or_more = [''.join(g) for g in letter_groups if len(g) >= 3]
    if len(triple_or_more) == 0:
        return word
    # replace letters to find the short word
    short_word = word
    for trip in triple_or_more:
        short_word = short_word.replace(trip, trip[0] * 2)

    return short_word


# Command line arguments are cast to bool type
def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


# decorator to time a function
def timer(function):
    def log_time():
        start_time = time.time()
        function()
        elapsed = time.time() - start_time
        logger.info('Function "{name}" finished in {time:.2f} s'.format(name=function.__name__, time=elapsed))
    return log_time()


# load embeddings from text file
def load_vectors(fname, vocab):
    fin = io.open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
    _, d = map(int, fin.readline().split())
    data = {}
    for line in fin:
        tokens = line.rstrip().split(' ')
        data[tokens[0]] = np.asarray(tokens[1:], dtype='float32')

    scale = 0.25
    # scale = np.sqrt(3.0 / n_dim)
    embedding_matrix = np.random.uniform(-scale, scale, [len(vocab), d])
    embedding_matrix[0] = np.zeros(d)
    cnt = 0
    for word, i in vocab._token2id.items():
        embedding_vector = data.get(word)
        if embedding_vector is not None:
            cnt += 1
            embedding_matrix[i] = embedding_vector
    logger.info('OOV rate: {:04.2f} %'.format(1 - cnt / len(vocab._token2id)))
    return embedding_matrix, d
