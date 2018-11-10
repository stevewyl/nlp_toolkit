"""
Sequence Labeler Wrapper
"""

from nlp_toolkit.model_zoo import Word_RNN, IDCNN
from nlp_toolkit.trainer import Trainer
from nlp_toolkit.utilities import logger


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
        self.mode = config['mode']
        if self.mode == 'train':
            assert config is not None
            self.config = config
            self.m_cfg = self.config[self.model_name]
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
                embeddings=self.config['word_embeddings'],
                inner_char=self.config['data']['inner_char'],
                use_crf=self.m_cfg['use_crf'],
                char_feature_method=self.m_cfg['char_feature_method'],
                integration_method=self.m_cfg['integration_method'],
                rnn_type=self.m_cfg['rnn_type'],
                nb_rnn_layers=self.m_cfg['nb_rnn_layers'],
                nb_filters=self.m_cfg['nb_filters'],
                conv_kernel_size=self.m_cfg['conv_kernel_size'],
                drop_rate=self.m_cfg['drop_rate'],
                re_drop_rate=self.m_cfg['re_drop_rate']
            )
        elif self.model_name == 'idcnn':
            model = IDCNN(
                nb_classes=self.config['nb_classes'],
                nb_tokens=self.config['nb_tokens'],
                maxlen=self.config['maxlen'],
                embedding_dim=self.config['embedding_dim'],
                embeddings=self.config['word_embeddings'],
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
            metric=t_cfg['metric']
        )
        return model_trainer

    def train(self, x, y):
        return self.model_trainer.train(
            x, y, self.transformer, self.seq_type)

    def predict(self, x):
        pass

    def evaluate(self, x, y):
        pass

    def load(self, weight_fname, para_fname):
        if self.model_name == 'word_rmm':
            self.model = Word_RNN.load(weight_fname, para_fname)
        else:
            pass
