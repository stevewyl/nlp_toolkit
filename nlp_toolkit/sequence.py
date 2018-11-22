"""
Text Sequence Utilties
"""

import math
import random
import numpy as np
from collections import Counter
from keras.utils import Sequence
from keras.utils.np_utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from sklearn.externals import joblib
from sklearn.base import BaseEstimator, TransformerMixin
from nlp_toolkit.utilities import timer, logger
from typing import Dict, List


def top_elements(array, k):
    ind = np.argpartition(array, -k)[-k:]
    return ind[np.argsort(array[ind])][::-1]


class Vocabulary(object):
    """
    Vocab Class for any NLP Tasks
    """

    def __init__(self, max_size=None, lower=True, unk_token=True, specials=('<pad>',)):
        self._max_size = max_size
        self._lower = lower
        self._unk = unk_token
        if specials:
            self._token2id = {token: i for i, token in enumerate(specials)}
            self._id2token = list(specials)
        else:
            self._token2id = {}
            self._id2token = []
        self._token_count = Counter()

    def __len__(self):
        return len(self._token2id)

    def add_token(self, token):
        token = self.process_token(token)
        self._token_count.update([token])

    def add_documents(self, docs):
        for sent in docs:
            sent = map(self.process_token, sent)
            self._token_count.update(sent)

    def doc2id(self, doc):
        # doc = map(self.process_token, doc)
        return [self.token_to_id(token) for token in doc]

    def id2doc(self, ids):
        return [self.id_to_token(idx) for idx in ids]

    def build(self):
        token_freq = self._token_count.most_common(self._max_size)
        idx = len(self.vocab)
        for token, _ in token_freq:
            self._token2id[token] = idx
            self._id2token.append(token)
            idx += 1
        if self._unk:
            unk = '<unk>'
            self._token2id[unk] = idx
            self._id2token.append(unk)

    def process_token(self, token):
        if self._lower:
            token = token.lower()

        return token

    def token_to_id(self, token):
        # token = self.process_token(token)
        return self._token2id.get(token, len(self._token2id) - 1)

    def id_to_token(self, idx):
        return self._id2token[idx]

    def extend_vocab(self, new_vocab, max_tokens=10000):
        assert isinstance(new_vocab, list)
        if max_tokens < 0:
            max_tokens = 10000
        base_index = self.__len__()
        added = 0
        for word in new_vocab:
            if added >= max_tokens:
                break
            if word not in self._token2id:
                self._token2id[word] = base_index + added
                self._id2token.append(word)
                added += 1
        logger.info('%d new words have been added to vocab' % added)
        return added

    @property
    def vocab(self):
        return self._token2id

    @property
    def reverse_vocab(self):
        return self._id2token


class IndexTransformer(BaseEstimator, TransformerMixin):
    """
    Similar with Sklearn function for transforming text to index
    Basic tokens are usually words.

    # Arguments:
        1. max_tokens: maximum number of basic tokens in one sentence
        2. max_inner_chars: maximum number of char tokens in one word
        3. lower: whether to lower tokers
        4. use_inner_char: whether to use inner char tokens depend on your model
        5. initial_vocab: the additional basic tokens which are not in corpus

    # Usage：
        p = IndexTransformer()
        new_data = p.fit_transform(data)
        # save
        p.save(file_name)
        # load
        p = IndexTransformer.load(file_name)
        # inverse transform y label
        y_true_label = p.inver_transform(y_pred)
    """

    def __init__(self, task_type, max_tokens=80, max_inner_chars=8, lower=True,
                 use_inner_char=False, initial_vocab=None,
                 use_seg=False, use_radical=False, basic_token='word'):
        self.basic_token = basic_token
        self.task_type = task_type
        self.max_tokens = max_tokens
        self.max_inner_chars = max_inner_chars
        self.use_inner_char = use_inner_char
        self.use_seg = use_seg
        self.use_radical = use_radical
        self._token_vocab = Vocabulary(lower=lower)
        self._label_vocab = Vocabulary(
            lower=False, unk_token=False, specials=None)
        if use_inner_char:
            self._inner_char_vocab = Vocabulary(lower=lower)
        if initial_vocab:
            self._token_vocab.add_documents([initial_vocab])
        if use_seg:
            self._seg_vocab = Vocabulary(lower=False)
        if use_radical:
            self._radical_vocab = Vocabulary(lower=False)

    def fit(self, X, y=None):
        assert isinstance(X, dict)
        self._token_vocab.add_documents(X['token'])
        self._token_vocab.build()
        if y is not None:
            self._label_vocab.add_documents(y)
            self._label_vocab.build()
        if self.use_inner_char:
            for doc in X['token']:
                self._inner_char_vocab.add_documents(doc)
            self._inner_char_vocab.build()
        if self.use_seg:
            self._seg_vocab.add_documents(X['seg'])
            self._seg_vocab.build()
        if self.use_radical:
            self._radical_vocab.add_documents(X['radical'])
            self._radical_vocab.build()

        return self

    def transform(self, X, y=None):
        tokens = X['token']
        token_ids = [self._token_vocab.doc2id(doc) for doc in tokens]
        lengths = (len(line) for line in token_ids)
        lengths = [self.max_tokens if l > self.max_tokens else l for l in lengths]
        token_ids = pad_sequences(
            token_ids, maxlen=self.max_tokens, padding='post')
        features = {'token': token_ids, 'length': lengths}

        if self.use_inner_char:
            char_ids = [[self._inner_char_vocab.doc2id(w) for w in doc] for doc in tokens]
            char_ids = pad_nested_sequences(
                char_ids, self.max_tokens, self.max_inner_chars)
            features['char'] = char_ids

        if self.use_seg:
            seg_ids = [self._seg_vocab.doc2id(doc) for doc in X['seg']]
            seg_ids = pad_sequences(
                seg_ids, maxlen=self.max_tokens, padding='post')
            features['seg'] = seg_ids

        if self.use_radical:
            radical_ids = [self._radical_vocab.doc2id(doc) for doc in X['radical']]
            radical_ids = pad_sequences(
                radical_ids, maxlen=self.max_tokens, padding='post')
            features['radical'] = radical_ids

        if y is not None:
            y = [self._label_vocab.doc2id(doc) for doc in y]
            if self.task_type == 'sequence_labeling':
                y = pad_sequences(y, maxlen=self.max_tokens, padding='post')
            y = to_categorical(y, self.label_size).astype(float)

            return features, y
        else:
            return features

    def fit_transform(self, X, y=None, **params):
        return self.fit(X, y).transform(X, y)

    def inverse_transform(self, y, lengths=None, top_k=1, return_percentage=False):
        if self.task_type == 'classification':
            if top_k == 1:
                ind_top = np.argmax(y, -1)
                inverse_y = [self._label_vocab.id2doc([idx])[0] for idx in ind_top]
                return inverse_y
            elif top_k > 1:
                ind_top = [top_elements(prob, top_k) for prob in y]
                inverse_y = [self._label_vocab.id2doc(id_list) for id_list in ind_top]
                if not return_percentage:
                    return inverse_y
                else:
                    pct_top = [[prob[ind] for ind in ind_top[idx]] for idx, prob in enumerate(y)]
                    return inverse_y, pct_top
        elif self.task_type == 'sequence_labeling':
            ind_top = np.argmax(y, -1)
            inverse_y = [self._label_vocab.id2doc(idx) for idx in ind_top]
            if lengths is not None:
                inverse_y = [iy[:l] for iy, l in zip(inverse_y, lengths)]
            return inverse_y

    @property
    def token_vocab_size(self):
        return len(self._token_vocab)

    @property
    def char_vocab_size(self):
        return len(self._inner_char_vocab)

    @property
    def seg_vocab_size(self):
        return len(self._seg_vocab)

    @property
    def radical_vocab_size(self):
        return len(self._radical_vocab)

    @property
    def label_size(self):
        return len(self._label_vocab)

    def save(self, file_path):
        joblib.dump(self, file_path)

    @classmethod
    def load(cls, file_path):
        p = joblib.load(file_path)
        # print('data transformer loaded')
        return p


def pad_nested_sequences(sequences, max_sent_len, max_word_len, dtype='int32'):
    """
    Pad char sequences of one single word
    """
    x = np.zeros((len(sequences), max_sent_len, max_word_len)).astype(dtype)
    for i, sent in enumerate(sequences):
        if len(sent) > max_sent_len:
            sent = sent[:max_sent_len]
        for j, word in enumerate(sent):
            if len(word) < max_word_len:
                x[i, j, :len(word)] = word
            else:
                x[i, j, :] = word[:max_word_len]
    return x


class BasicIterator(Sequence):
    """
    Wrapper for Keras Sequence Class
    """

    def __init__(self, x, y=None, batch_size=1,
                 extra_features: List[str] = None, concat=False):
        self.x = x
        self.y = y
        self.batch_size = batch_size
        self.concat = concat
        self.extra_features = extra_features

    def __getitem__(self, idx):
        idx_begin = self.batch_size * idx
        idx_end = self.batch_size * (idx + 1)
        x_b = self.x[idx_begin: idx_end]
        if self.concat:
            x_word = x_b[:, :, 0]
            x_char = x_b[:, :, 1:]
            batch_x = {'token': x_word, 'char': x_char}
        elif self.extra_features:
            batch_x = {'token': x_b[:, :, 0]}
            for k, feature in enumerate(self.extra_features):
                batch_x[feature] = x_b[:, :, k+1]
        else:
            batch_x = {'token': x_b}
        if self.y is not None:
            batch_y = self.y[idx_begin: idx_end]
            return batch_x, batch_y
        else:
            return batch_x

    def __len__(self):
        return math.ceil(len(self.x) / self.batch_size)


def _roundto(val, batch_size):
    return int(math.ceil(val / batch_size)) * batch_size


class BucketIterator(Sequence):
    """
    A Keras Sequence (dataset reader) of input sequences read in bucketed bins.
    Assumes all inputs are already padded using 'pad_sequences'
    (where post padding is prepended).
    """

    def __init__(self, task_type, num_buckets, batch_size, seq_lengths,
                 x_seq, y=None, extra_features: List[str] = None, concat=False):
        cnt_labels = y.shape[-1]
        self.batch_size = batch_size
        self.concat = concat
        self.task_type = task_type
        self.extra_features = extra_features
        # Count bucket sizes
        bucket_sizes, bucket_ranges = np.histogram(seq_lengths,
                                                   bins=num_buckets)

        # Looking for non-empty buckets
        actual_buckets = [bucket_ranges[i+1]
                          for i, bs in enumerate(bucket_sizes) if bs > 0]
        actual_bucket_sizes = [bs for bs in bucket_sizes if bs > 0]
        bucket_seqlen = [int(math.ceil(bs)) for bs in actual_buckets]
        num_actual = len(actual_buckets)
        logger.info('Training with %d non-empty buckets' % num_actual)

        if task_type == 'sequence_labeling':
            if concat or extra_features:
                self.bins = [(np.ndarray([bs, bsl, x_seq.shape[-1]], dtype=x_seq.dtype),
                              np.ndarray([bs, bsl, cnt_labels], dtype=y.dtype))
                             for bsl, bs in zip(bucket_seqlen, actual_bucket_sizes)]
            else:
                self.bins = [(np.ndarray([bs, bsl], dtype=x_seq.dtype),
                              np.ndarray([bs, bsl, cnt_labels], dtype=y.dtype))
                             for bsl, bs in zip(bucket_seqlen, actual_bucket_sizes)]
        elif task_type == 'classification':
            self.bins = [(np.ndarray([bs, bsl], dtype=x_seq.dtype),
                          np.ndarray([bs, cnt_labels], dtype=y.dtype))
                         for bsl, bs in zip(bucket_seqlen, actual_bucket_sizes)]
        assert len(self.bins) == num_actual

        # Insert the sequences into the bins
        bctr = [0] * num_actual
        for i, sl in enumerate(seq_lengths):
            for j in range(num_actual):
                bsl = bucket_seqlen[j]
                if sl < bsl or j == num_actual - 1:
                    if self.task_type == 'sequence_labeling':
                        self.bins[j][1][bctr[j], :bsl, :] = y[i, :bsl, :]
                        if self.concat or self.extra_features:
                            self.bins[j][0][bctr[j], :bsl, :] = x_seq[i, :bsl, :]
                        else:
                            self.bins[j][0][bctr[j], :bsl] = x_seq[i, :bsl]
                    elif task_type == 'classification':
                        self.bins[j][0][bctr[j], :bsl] = x_seq[i, :bsl]
                        self.bins[j][1][bctr[j], :] = y[i]
                    bctr[j] += 1
                    break

        self.num_samples = x_seq.shape[0]
        self.dataset_len = int(sum([math.ceil(bs / self.batch_size)
                                    for bs in actual_bucket_sizes]))
        self._permute()

    def _permute(self):
        # Shuffle bins
        random.shuffle(self.bins)

        # Shuffle bin contents
        for i, (xbin, ybin) in enumerate(self.bins):
            index_array = np.random.permutation(xbin.shape[0])
            self.bins[i] = (xbin[index_array], ybin[index_array])

    def on_epoch_end(self):
        self._permute()

    def __len__(self):
        return self.dataset_len

    def __getitem__(self, idx):
        idx_begin = self.batch_size * idx
        idx_end = self.batch_size * (idx + 1)

        # Obtain bin index
        for _, (xbin, ybin) in enumerate(self.bins):
            rounded_bin = _roundto(xbin.shape[0], self.batch_size)
            if idx_begin >= rounded_bin:
                idx_begin -= rounded_bin
                idx_end -= rounded_bin
                continue

            # Found bin
            idx_end = min(xbin.shape[0], idx_end)  # Clamp to end of bin
            x_b = xbin[idx_begin:idx_end]
            if self.concat:
                x_word = x_b[:, :, 0]
                x_char = x_b[:, :, 1:]
                batch_x = {'token': x_word, 'char': x_char}
            elif self.extra_features:
                batch_x = {}
                batch_x['token'] = x_b[:, :, 0]
                for k, feature in enumerate(self.extra_features):
                    batch_x[feature] = x_b[:, :, k+1]
            else:
                batch_x = {'token': x_b}
            batch_y = ybin[idx_begin:idx_end]
            return batch_x, batch_y

        raise ValueError('out of bounds')
