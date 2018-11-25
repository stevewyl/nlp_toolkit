from nlp_toolkit.models import Base_Model
from nlp_toolkit.modules.token_embedders import Token_Embedding
from nlp_toolkit.modules.logits import sl_output_logits
from keras.layers import Input, BatchNormalization
from keras.layers import LSTM, GRU, Bidirectional
from keras.layers.merge import concatenate
from keras.models import Model
import sys


class Char_RNN(Base_Model):
    """
    Similar model structure to Word_RNN. But use char as basic token.
    And some useful features are included: 1. radicals 2. segmentation tag 3. nchar
    """

    def __init__(self, nb_classes, nb_tokens, maxlen,
                 embedding_dim=64, use_crf=True,
                 use_seg=False, use_radical=False,
                 use_nchar=False,
                 nb_seg_tokens=None, nb_radical_tokens=None,
                 rnn_type='lstm', nb_rnn_layers=2,
                 char_rnn_size=128, drop_rate=0.5,
                 re_drop_rate=0.15, embed_dropout_rate=0.15):
        self.nb_classes = nb_classes
        self.nb_tokens = nb_tokens
        self.maxlen = maxlen
        self.embedding_dim = embedding_dim
        self.use_crf = use_crf
        self.use_seg = use_seg
        self.use_radical = use_radical
        self.use_nchar = False
        self.rnn_type = rnn_type
        self.nb_rnn_layers = nb_rnn_layers
        self.drop_rate = drop_rate
        self.re_drop_rate = re_drop_rate
        self.char_rnn_size = char_rnn_size
        self.embed_dropout_rate = embed_dropout_rate
        if use_seg:
            self.nb_seg_tokens = nb_seg_tokens
        if use_radical:
            self.nb_radical_tokens = nb_radical_tokens

        self.invalid_params = {}
        super(Char_RNN).__init__()

    def forward(self):
        char_ids = Input(shape=(self.maxlen,), dtype='int32', name='token')
        input_data = [char_ids]
        char_embed = Token_Embedding(
            char_ids, self.nb_tokens,
            self.embedding_dim, None, True,
            self.maxlen, self.embed_dropout_rate, name='char_embeddings')
        embed_features = [char_embed]
        if self.use_seg:
            seg_ids = Input(shape=(self.maxlen,), dtype='int32', name='seg')
            input_data.append(seg_ids)
            seg_emebd = Token_Embedding(
                seg_ids, self.nb_seg_tokens, 8, None, True,
                self.maxlen, name='seg_embeddings')
            embed_features.append(seg_emebd)
        if self.use_radical:
            radical_ids = Input(shape=(self.maxlen,), dtype='int32', name='radical')
            input_data.append(radical_ids)
            radical_embed = Token_Embedding(
                radical_ids, self.nb_radical_tokens, 32,
                None, True, self.maxlen, name='radical_embeddings')
            embed_features.append(radical_embed)
        if self.use_nchar:
            pass
        if self.use_seg or self.use_radical:
            x = concatenate(embed_features, axis=-1, name='embed')
        else:
            x = char_embed
        x = BatchNormalization()(x)

        for i in range(self.nb_rnn_layers):
            if self.rnn_type == 'lstm':
                x = Bidirectional(
                    LSTM(self.char_rnn_size, dropout=self.drop_rate,
                         recurrent_dropout=self.re_drop_rate,
                         return_sequences=True), name='char_lstm_%d' % (i+1))(x)
            elif self.rnn_type == 'gru':
                x = Bidirectional(
                    GRU(self.char_rnn_size, dropout=self.drop_rate,
                        recurrent_dropout=self.re_drop_rate,
                        return_sequences=True), name='char_gru_%d' % (i+1))(x)
            else:
                print('invalid rnn type, only support lstm and gru')
                sys.exit()

        outputs, self._loss, self._acc = sl_output_logits(
            x, self.nb_classes, self.use_crf)
        self.model = Model(inputs=input_data, outputs=outputs)

    def get_loss(self):
        return self._loss

    def get_metrics(self):
        return self._acc
