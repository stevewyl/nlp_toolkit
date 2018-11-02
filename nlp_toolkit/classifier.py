"""
Classifier Wrapper
"""

from nlp_toolkit.model_zoo import bi_lstm_attention
from nlp_toolkit.model_zoo import multi_head_self_attention
from nlp_toolkit.trainer import Trainer
from nlp_toolkit.utilities import logger


class Classifier(object):
    """
    Classifier Model Zoos. Include following models:

    1. TextCNN
    2. DPCNN
    3. Bi-GRU
    4. Bi-LSTM-Attention
    5. Multi-Head-Self-Attention
    """

    def __init__(self, model_name, config):
        self.model_name = model_name
        self.config = config
        self.model = self.get_model()
        self.model_trainer = self.get_trainer()

    def get_model(self):
        m_cfg = self.config[self.model_name]
        if self.model_name == 'bi_lstm_att':
            model = bi_lstm_attention(
                nb_classes=m_cfg['nb_classes'],
                nb_tokens=m_cfg['nb_tokens'],
                maxlen=m_cfg['maxlen'],
                embed_dim=m_cfg['embed_dim'],
                embeddings=m_cfg['embeddings'],
                final_dropout_rate=m_cfg['f_drop_rate'],
                embed_dropout_rate=m_cfg['e_drop_rate'],
                return_attention=m_cfg['return_att']
            )
        elif self.model_name == 'multi_head_self_att':
            model = multi_head_self_attention(
                nb_classes=m_cfg['nb_classes'],
                nb_tokens=m_cfg['nb_tokens'],
                maxlen=m_cfg['maxlen'],
                embed_dim=m_cfg['embed_dim'],
                embeddings=m_cfg['embeddings'],
                pos_embed=m_cfg['pos_embed']
            )
        else:
            logger.error('The model name ' + self.model_name + ' is unknown')
        return model

    def get_trainer(self):
        t_cfg = self.config['train']
        model_trainer = Trainer(
            self.model,
            batch_size=t_cfg['batch_size'],
            max_epoch=t_cfg['epoch'],
            model_name=self.model_name,
            train_mode=t_cfg['train_mode'],
            fold_cnt=t_cfg['n_fold'],
            test_size=t_cfg['test_size']
        )
        return model_trainer

    def train(self, x, y, transformer, seq='bucket'):
        return self.model_trainer.train(x, y, transformer, seq)

    def predict(self, x):
        pass

    def load(self, model_name, weight_fname, para_fname):
        if model_name == 'bi_lstm_att':
            self.model = bi_lstm_attention.load(weight_fname, para_fname)
        elif model_name == 'multi_head_self_att':
            self.model = multi_head_self_attention.load(weight_fname, para_fname)
