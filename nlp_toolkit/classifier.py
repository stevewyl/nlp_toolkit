"""
Classifier Wrapper
"""

import sys
import time
from nlp_toolkit.models import bi_lstm_attention
from nlp_toolkit.models import Transformer
from nlp_toolkit.models import textCNN, DPCNN
from nlp_toolkit.trainer import Trainer
from nlp_toolkit.utilities import logger
from nlp_toolkit.sequence import BasicIterator


# TODO
# 1. evaluate func
class Classifier(object):
    """
    Classifier Model Zoos. Include following models:

    1. TextCNN
    2. DPCNN (Deep Pyramid CNN)
    3. Bi-LSTM-Attention
    4. Multi-Head-Self-Attention (Transformer)
    5. HAN (Hierachical Attention Network)
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
        if self.model_name == 'bi_lstm_att':
            model = bi_lstm_attention(
                nb_classes=self.config['nb_classes'],
                nb_tokens=self.config['nb_tokens'],
                maxlen=self.config['maxlen'],
                embedding_dim=self.config['embedding_dim'],
                embeddings=self.config['token_embeddings'],
                rnn_size=self.m_cfg['rnn_size'],
                attention_dim=self.m_cfg['attention_dim'],
                final_dropout_rate=self.m_cfg['final_drop_rate'],
                embed_dropout_rate=self.m_cfg['embed_drop_rate'],
                return_attention=self.m_cfg['return_att']
            )
        elif self.model_name == 'transformer':
            model = Transformer(
                nb_classes=self.config['nb_classes'],
                nb_tokens=self.config['nb_tokens'],
                maxlen=self.config['maxlen'],
                embedding_dim=self.config['embedding_dim'],
                embeddings=self.config['token_embeddings'],
                pos_embed=self.m_cfg['pos_embed'],
                nb_transformer=self.m_cfg['nb_transformer'],
                final_dropout_rate=self.m_cfg['final_drop_rate'],
                embed_dropout_rate=self.m_cfg['embed_drop_rate']
            )
        elif self.model_name == 'text_cnn':
            model = textCNN(
                nb_classes=self.config['nb_classes'],
                nb_tokens=self.config['nb_tokens'],
                maxlen=self.config['maxlen'],
                embedding_dim=self.config['embedding_dim'],
                embeddings=self.config['token_embeddings'],
                conv_kernel_size=self.m_cfg['conv_kernel_size'],
                pool_size=self.m_cfg['pool_size'],
                nb_filters=self.m_cfg['nb_filters'],
                fc_size=self.m_cfg['fc_size'],
                embed_dropout_rate=self.m_cfg['embed_drop_rate']
            )
        elif self.model_name == 'dpcnn':
            model = DPCNN(
                nb_classes=self.config['nb_classes'],
                nb_tokens=self.config['nb_tokens'],
                maxlen=self.config['maxlen'],
                embedding_dim=self.config['embedding_dim'],
                embeddings=self.config['token_embeddings'],
                region_kernel_size=self.m_cfg['region_kernel_size'],
                conv_kernel_size=self.m_cfg['conv_kernel_size'],
                pool_size=self.m_cfg['pool_size'],
                nb_filters=self.m_cfg['nb_filters'],
                repeat_time=self.m_cfg['repeat_time'],
                final_dropout_rate=self.m_cfg['final_drop_rate'],
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
            patiences=t_cfg['patiences'],
            extra_features=self.extra_features
        )
        return model_trainer

    def train(self, x, y):
        if self.model_name == 'bi_lstm_att':
            return_att = self.m_cfg['return_att']
        else:
            return_att = False
        return self.model_trainer.train(
            x, y, self.transformer, self.seq_type, return_att)

    def predict(self, x, batch_size=64, return_attention=False):
        x = x['token']
        n_labels = len(self.transformer._label_vocab._id2token)
        start = time.time()
        x_seq = BasicIterator(x, batch_size=batch_size)
        result = self.model.model.predict_generator(x_seq)
        y_pred = self.transformer.inverse_transform(result[:, :n_labels])
        used_time = time.time() - start
        logger.info('predict {} samples used {:4.1f}s'.format(
            len(x), used_time))
        if result.shape[1] > n_labels:
            attention = result[:, n_labels:]
            return y_pred, attention
        else:
            return y_pred

    def evaluate(self, x, y):
        pass

    def load(self, weight_fname, para_fname):
        if self.model_name == 'bi_lstm_att':
            self.model = bi_lstm_attention.load(weight_fname, para_fname)
        elif self.model_name == 'multi_head_self_att':
            self.model = Transformer.load(weight_fname, para_fname)
        elif self.model_name == 'text_cnn':
            self.model = textCNN.load(weight_fname, para_fname)
        elif self.model_name == 'dpcnn':
            self.model = DPCNN.load(weight_fname, para_fname)
        else:
            logger.warning('invalid model name')
            sys.exit()
