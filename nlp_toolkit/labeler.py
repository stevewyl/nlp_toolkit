"""
Sequence Labeler Wrapper
"""

import sys
import time
import numpy as np
from copy import deepcopy
from nlp_toolkit.models import Word_RNN, IDCNN, Char_RNN
from nlp_toolkit.trainer import Trainer
from nlp_toolkit.utilities import logger
from nlp_toolkit.sequence import BasicIterator
from nlp_toolkit.data import Dataset
from typing import List


class Labeler(object):
    """
    Sequence Labeling Model Zoos. Include following models:

    1. WordRNN + Inner_Char
    2. CharRNN + Extra Embeddings (segment, radical, nchar)
    3. IDCNN
    """

    def __init__(self, model_name, dataset: Dataset, seq_type='bucket'):
        self.model_name = model_name
        self.dataset = dataset
        self.transformer = dataset.transformer
        self.config = self.dataset.config
        if self.config['mode'] == 'train':
            self.m_cfg = self.config['model'][self.model_name]
            self.seq_type = seq_type
            if seq_type == 'bucket':
                self.config['maxlen'] = None
            self.model = self.get_model()
            self.model_trainer = self.get_trainer()
        elif self.config['mode'] == 'predict':
            pass
        else:
            logger.warning('invalid mode name. Current only support "train" and "predict"')

    def get_model(self):
        if self.model_name == 'word_rnn':
            model = Word_RNN(
                nb_classes=self.config['nb_classes'],
                nb_tokens=self.config['nb_tokens'],
                nb_char_tokens=self.config['nb_char_tokens'],
                maxlen=self.config['maxlen'],
                embedding_dim=self.config['embedding_dim'],
                embeddings=self.config['token_embeddings'],
                inner_char=self.config['data']['inner_char'],
                use_crf=self.m_cfg['use_crf'],
                char_feature_method=self.m_cfg['char_feature_method'],
                integration_method=self.m_cfg['integration_method'],
                rnn_type=self.m_cfg['rnn_type'],
                nb_rnn_layers=self.m_cfg['nb_rnn_layers'],
                nb_filters=self.m_cfg['nb_filters'],
                conv_kernel_size=self.m_cfg['conv_kernel_size'],
                drop_rate=self.m_cfg['drop_rate'],
                re_drop_rate=self.m_cfg['re_drop_rate'],
                word_rnn_size=self.m_cfg['word_rnn_size'],
                embed_dropout_rate=self.m_cfg['embed_drop_rate']
            )
        elif self.model_name == 'char_rnn':
            model = Char_RNN(
                nb_classes=self.config['nb_classes'],
                nb_tokens=self.config['nb_tokens'],
                nb_seg_tokens=self.config['nb_seg_tokens'],
                nb_radical_tokens=self.config['nb_radical_tokens'],
                maxlen=self.config['maxlen'],
                embedding_dim=self.config['embedding_dim'],
                use_seg=self.config['use_seg'],
                use_radical=self.config['use_radical'],
                use_crf=self.m_cfg['use_crf'],
                rnn_type=self.m_cfg['rnn_type'],
                nb_rnn_layers=self.m_cfg['nb_rnn_layers'],
                drop_rate=self.m_cfg['drop_rate'],
                re_drop_rate=self.m_cfg['re_drop_rate'],
                char_rnn_size=self.m_cfg['char_rnn_size'],
                embed_dropout_rate=self.m_cfg['embed_drop_rate']
            )
        elif self.model_name == 'idcnn':
            model = IDCNN(
                nb_classes=self.config['nb_classes'],
                nb_tokens=self.config['nb_tokens'],
                maxlen=self.config['maxlen'],
                embedding_dim=self.config['embedding_dim'],
                embeddings=self.config['token_embeddings'],
                use_crf=self.m_cfg['use_crf'],
                nb_filters=self.m_cfg['nb_filters'],
                conv_kernel_size=self.m_cfg['conv_kernel_size'],
                drop_rate=self.m_cfg['drop_rate'],
                repeat_times=self.m_cfg['repeat_times'],
                dilation_rate=self.m_cfg['dilation_rate'],
                embed_dropout_rate=self.m_cfg['embed_drop_rate']
            )
        else:
            logger.warning('The model name ' + self.model_name + ' is unknown')
            model = None
        return model

    def get_trainer(self):
        t_cfg = self.config['train']
        model_trainer = Trainer(
            self.model,
            model_name=self.model_name,
            task_type=self.config['task_type'],
            batch_size=t_cfg['batch_size'],
            max_epoch=t_cfg['epochs'],
            train_mode=t_cfg['train_mode'],
            fold_cnt=t_cfg['nb_fold'],
            test_size=t_cfg['test_size'],
            metric=t_cfg['metric'],
            nb_bucket=t_cfg['nb_bucket'],
            patiences=t_cfg['patiences']
        )
        return model_trainer

    def train(self):
        return self.model_trainer.train(
            self.dataset.texts, self.dataset.labels,
            self.transformer, self.seq_type)

    def feature_concat(self, x, features):
        x_ori = deepcopy(x)
        x = np.expand_dims(x['token'], axis=-1)
        for name in features:
            feature = np.expand_dims(x_ori[name], axis=-1)
            x = np.concatenate((x, feature), axis=-1)
        return x

    def predict(self, x: Dict[str, List[List[str]]], batch_size=64,
                return_prob=False):
        start = time.time()
        x_seq = BasicIterator('sequence_labeling', self.transformer,
                              x, batch_size=batch_size)
        result = self.model.model.predict_generator(x_seq)
        if return_prob:
            y_pred = result
        else:
            y_pred = self.transformer.inverse_transform(result, lengths=lengths)
        used_time = time.time() - start
        logger.info('predict {} samples used {:4.1f}s'.format(
            len(x), used_time))
        return y_pred

    def show_results(self, x, y_pred):
        x_len = x['length']
        y_pred_true = [y_pred[i][:x_len[i]] for i in range(len(x_len))]
        x_true = [x['token'][i][:x_len[i]] for i in range(len(x_len))]
        x_true = [[self.transformer._token_vocab.id_to_token(idx) for idx in seq] for seq in x_true]
        return [[(x1, y1) for x1, y1 in zip(x, y)] for x, y in zip(x_true, y_pred_true)]

    def evaluate(self, x: Dict[str, List[List[str]]], y: List[List[str]]):
        pass

    def load(self, weight_fname, para_fname):
        if self.model_name == 'word_rnn':
            self.model = Word_RNN.load(weight_fname, para_fname)
        elif self.model_name == 'char_rnn':
            self.model = Char_RNN.load(weight_fname, para_fname)
        elif self.model_name == 'idcnn':
            self.model = IDCNN.load(weight_fname, para_fname)
        else:
            logger.warning('invalid model name')
            sys.exit()
