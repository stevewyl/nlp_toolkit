"""
Sequence Labeler Wrapper
"""

import sys
import time
import numpy as np
from nlp_toolkit.model_zoo import Word_RNN, IDCNN, Char_RNN
from nlp_toolkit.trainer import Trainer
from nlp_toolkit.utilities import logger
from nlp_toolkit.sequence import BasicIterator


class Labeler(object):
    """
    Sequence Labeling Model Zoos. Include following models:

    1. WordRNN + Inner_Char
    2. CharRNN + Extra Embeddings (segment, radical, nchar)
    3. IDCNN
    """

    def __init__(self, model_name, transformer, seq_type='bucket', config=None):
        self.model_name = model_name
        self.transformer = transformer
        if config:
            self.mode = config['mode']
            self.extra_features = config['extra_features']
        else:
            self.mode = 'predict'
        if self.mode == 'train':
            assert config is not None
            self.config = config
            self.m_cfg = self.config['model'][self.model_name]
            self.seq_type = seq_type
            if seq_type == 'bucket':
                self.config['maxlen'] = None
            self.model = self.get_model()
            self.model_trainer = self.get_trainer()
        elif self.mode == 'predict':
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
                word_rnn_size=self.m_cfg['word_rnn_size']
            )
        elif self.model_name == 'char_rnn':
            model = Char_RNN(
                nb_classes=self.config['nb_classes'],
                nb_tokens=self.config['nb_tokens'],
                nb_seg_tokens=self.config['nb_seg_tokens'],
                nb_radical_tokens=self.config['nb_radical_tokens'],
                maxlen=self.config['maxlen'],
                emebdding_dim=self.config['embedding_dim'],
                use_seg=self.config['use_seg'],
                use_radical=self.config['use_radical'],
                use_crf=self.m_cfg['use_crf'],
                rnn_type=self.m_cfg['rnn_type'],
                nb_rnn_layers=self.m_cfg['nb_rnn_layers'],
                drop_rate=self.m_cfg['drop_rate'],
                re_drop_rate=self.m_cfg['re_drop_rate'],
                char_rnn_size=self.m_cfg['char_rnn_size']
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
                dilation_rate=self.m_cfg['dilation_rate']
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
            patiences=t_cfg['patiences'],
            extra_features=self.extra_features
        )
        return model_trainer

    def train(self, x, y):
        return self.model_trainer.train(
            x, y, self.transformer, self.seq_type)

    def predict(self, x, batch_size=64):
        use_inner_char = self.transformer.use_inner_char
        use_seg = self.transformer.use_seg
        use_radical = self.transformer.use_radical
        if use_inner_char or use_seg or use_radical:
            x_word = x['token']
            x_word = np.expand_dims(x_word, axis=-1)
            if use_inner_char:
                concat = True
                x = np.concatenate((x_word, x['inner_char']), axis=-1)
            else:
                concat = False
                extra_features = []
                if use_seg:
                    x = np.concatenate((x_word, x['seg']), axis=-1)
                    extra_features.append('seg')
                if use_radical:
                    x = np.concatenate((x_word, x['radical']), axis=-1)
                    extra_features.append('radical')
        else:
            concat = False
            x = x['token']

        start = time.time()
        x_seq = BasicIterator(x, batch_size=batch_size,
                              extra_features=extra_features,
                              concat=concat)
        result = self.model.model.predict_generator(x_seq)
        y_pred = self.transformer.inverse_transform(result)
        used_time = time.time() - start
        logger.info('predict {} samples used {:4.1f}s'.format(
            len(x), used_time))
        return y_pred

    def evaluate(self, x, y):
        pass

    def load(self, weight_fname, para_fname):
        if self.model_name == 'word_rnn':
            self.model = Word_RNN.load(weight_fname, para_fname)
        else:
            logger.warning('invalid model name')
            sys.exit()
