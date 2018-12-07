"""
some nlp process utilty functions
"""

import io
import re
import sys
import time
import logging
import numpy as np
from itertools import groupby

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s')
logger = logging.getLogger('nlp_toolkit')

global special_tokens
special_tokens = set(['s_', 'lan_', 'ss_'])


# [1, ['a', 'b], [True, False]] ---> [1, 'a', 'b', True, False]
def flatten_gen(x):
    for i in x:
        if isinstance(i, list) or isinstance(i, tuple):
            for inner_i in i:
                yield inner_i
        else:
            yield i


# judge char type ['cn', 'en', 'num', 'other']
def char_type(word):
    for char in word:
        unicode_char = ord(char)
        if unicode_char >= 19968 and unicode_char <= 40869:
            yield (char, 'cn')
        elif unicode_char >= 65 and unicode_char <= 122:
            yield (char, 'en')
        elif unicode_char >= 48 and unicode_char <= 57:
            yield (char, 'num')
        else:
            yield (char, 'other')


# split word into chars
def split_cn_en(word):
    new_word = [c for c in char_type(word)]
    new_word_len = len(new_word)
    tmp = ''
    for ix, item in enumerate(new_word):
        if item[1] in {'en', 'num'}:
            if ix < new_word_len - 1:
                if new_word[ix+1][1] == item[1]:
                    tmp += item[0]
                else:
                    tmp += item[0]
                    yield tmp
                    tmp = ''
            else:
                tmp += item[0]
                yield tmp
        else:
            yield item[0]


# reassign token labels according new tokens
def extract_char(word_list, label_list=None, use_seg=False):
    if label_list:
        for word, label in zip(word_list, label_list):
            # label = label.strip('#')
            single_check = word in special_tokens or not re.search(r'[^a-z0-9]+', word)
            if len(word) == 1 or single_check:
                if use_seg:
                    yield (word, label, 'S')
                else:
                    yield (word, label)
            else:
                try:
                    new_word = list(split_cn_en(word))
                    word_len = len(new_word)
                    if label == 'O':
                        new_label = ['O'] * word_len
                    elif label.startswith('I'):
                        new_label = [label] * word_len
                    else:
                        label_i = 'I' + label[1:]
                        if label.startswith('B'):
                            new_label = [label] + [label_i] * (word_len - 1)
                        elif label.startswith('E'):
                            new_label = [label_i] * (word_len - 1) + [label]
                    if use_seg:
                        seg_tag = ['M'] * word_len
                        seg_tag[0] = 'B'
                        seg_tag[-1] = 'E'
                        for x, y, z in zip(new_word, new_label, seg_tag):
                            yield (x, y, z)
                    else:
                        for x, y in zip(new_word, new_label):
                            yield (x, y)
                except Exception as e:
                    print(e)
                    print(list(zip(word_list, label_list)))
                    sys.exit()
    else:
        for word in word_list:
            single_check = word in special_tokens or not re.search(r'[^a-z0-9]+', word)
            if len(word) == 1 or single_check:
                if use_seg:
                    yield (word, 'S')
                else:
                    yield (word)
            else:
                new_word = list(split_cn_en(word))
                if use_seg:
                    seg_tag = ['M'] * len(new_word)
                    seg_tag[0] = 'B'
                    seg_tag[-1] = 'E'
                    for x, y in zip(new_word, seg_tag):
                        yield (x, y)
                else:
                    for x in new_word:
                        yield x


# get radical token by chars
def get_radical(d, char_list):
    return [d[char] if char in d else '<unk>' for char in char_list]


def word2char(word_list, label_list=None, task_type='',
              use_seg=False, use_radical=False, radical_dict=None):
    """
    convert basic token from word to char
    """

    if task_type == 'classification':
        assert label_list is None
        assert use_radical is False
        assert use_seg is False
        return [char for word in word_list for char in list(split_cn_en(word))]
    elif task_type == 'sequence_labeling':
        results = list(
            zip(*[item for item in extract_char(word_list, label_list, use_seg)]))
        if label_list:
            if use_seg:
                chars, new_labels, seg_tags = results
                assert len(chars) == len(new_labels) == len(seg_tags)
            else:
                chars, new_labels = results
                assert len(chars) == len(new_labels)
            new_result = {'token': chars, 'label': new_labels}
        else:
            if use_seg:
                chars, seg_tags = results
                assert len(chars) == len(seg_tags)
            else:
                chars = results
            new_result = {'token': chars}
        if use_seg:
            new_result['seg'] = seg_tags
        if use_radical:
            new_result['radical'] = get_radical(radical_dict, chars)
        return new_result
    else:
        logger.error('invalid task type')
        sys.exit()


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


# generate small embedding files according given vocabs
def gen_small_embedding(vocab_file, embed_file, output_file):
    vocab = set([word.strip() for word in open(vocab_file, encoding='utf8')])
    print('total vocab: ', len(vocab))
    fin = io.open(embed_file, 'r', encoding='utf-8', newline='\n', errors='ignore')
    try:
        n, d = map(int, fin.readline().split())
    except Exception:
        print('please make sure the embed file is gensim-formatted')

    def gen():
        for line in fin:
            token = line.rstrip().split(' ', 1)[0]
            if token in vocab:
                yield line

    result = [line for line in gen()]
    rate = 1 - len(result) / len(vocab)
    print('oov rate: {:4.2f}%'.format(rate * 100))

    with open(output_file, 'w', encoding='utf8') as fout:
        fout.write(str(len(result)) + ' ' + str(d) + '\n')
        for line in result:
            fout.write(line)


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


def load_tc_data(fname, label_prefix='__label__', max_tokens_per_doc=-1):

    def gen():
        with open(fname, 'r', encoding='utf8') as fin:
            for line in fin:
                words = line.strip().split()
                if words:
                    nb_labels = 0
                    label_line = []
                    for word in words:
                        if word.startswith(label_prefix):
                            nb_labels += 1
                            label = word.replace(label_prefix, "")
                            label_line.append(label)
                        else:
                            break
                    text = words[nb_labels:]
                    if len(text) > max_tokens_per_doc:
                        text = text[:max_tokens_per_doc]
                    yield (text, label_line)

    texts, labels = zip(*[item for item in gen()])
    return texts, labels


def load_sl_data(fname, data_format='basic'):

    def process_conll(data):
        tokens, tags = [], []
        for line in data:
            if line:
                token, tag = line.split('\t')
                tokens.append(token)
                tags.append(tag)
            else:
                yield (tokens, tags)
                tokens, tags = [], []

    data = (line.strip() for line in open(fname, 'r', encoding='utf8'))
    if data_format:
        if data_format == 'basic':
            texts, labels = zip(
                *[zip(*[item.rsplit('###', 1) for item in line.split('\t')]) for line in data])
        elif data_format == 'conll':
            texts, labels = zip(*[item for item in process_conll(data)])
        return texts, labels
    else:
        print('invalid data format for sequence labeling task')


def convert_seq_format(fin_name, fout_name, dest_format='conll'):
    if dest_format == 'conll':
        basic2conll(fin_name, fout_name)
    elif dest_format == 'basic':
        conll2basic(fin_name, fout_name)
    else:
        logger.warning('invalid data format')


def basic2conll(fin_name, fout_name):
    data = [line.strip() for line in open(fin_name, 'r', encoding='utf8')]
    with open(fout_name, 'w', encoding='utf8') as fout:
        for line in data:
            for item in line.split('\t'):
                token, label = item.rsplit('###')
                label = label.strip('#')
                fout.write(token + '\t' + label + '\n')
            fout.write('\n')


def conll2basic(fin_name, fout_name):
    data = [line.strip() for line in open(fin_name, 'r', encoding='utf8')]
    with open(fout_name, 'w', encoding='utf8') as fout:
        tmp = []
        for line in data:
            if line:
                token, label = line.split('\t')
                label = label.strip('\t')
                item = token + '###' + label
                tmp.append(item)
            else:
                new_line = '\t'.join(tmp) + '\n'
                fout.write(new_line)
                tmp = []
