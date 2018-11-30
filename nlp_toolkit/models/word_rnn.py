from nlp_toolkit.models import Base_Model
from nlp_toolkit.modules.token_embedders import Token_Embedding
from nlp_toolkit.modules.logits import sl_output_logits
from keras.layers import Input, Activation, TimeDistributed, Dense
from keras.layers import LSTM, GRU, Bidirectional
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras.layers import subtract, multiply, add, Lambda
from keras.layers.merge import concatenate
from keras.models import Model
import keras.backend as K
import sys


class Word_RNN(Base_Model):
    """
    Baseline sequence labeling model. Basic token is word.
    Support following extensibility:
        1. Extract inner-char features by using lstm or cnn
        2. Concat or attention between word and char features
    """

    def __init__(self, nb_classes, nb_tokens, maxlen,
                 nb_char_tokens=None, max_charlen=10,
                 embedding_dim=128, char_embedding_dim=32,
                 word_rnn_size=128, char_rnn_size=32,
                 embeddings=None, char_embeddings=None,
                 inner_char=False, use_crf=True,
                 char_feature_method='rnn',
                 integration_method='concat',
                 rnn_type='lstm',
                 nb_rnn_layers=1,
                 nb_filters=32,
                 conv_kernel_size=2,
                 drop_rate=0.5,
                 re_drop_rate=0.15,
                 embed_l2=1E-6,
                 embed_dropout_rate=0.15):
        super(Word_RNN).__init__()
        self.nb_classes = nb_classes
        self.nb_tokens = nb_tokens
        self.maxlen = maxlen
        self.embedding_dim = embedding_dim
        self.rnn_type = rnn_type
        self.nb_rnn_layers = nb_rnn_layers
        self.drop_rate = drop_rate
        self.re_drop_rate = re_drop_rate
        self.use_crf = use_crf
        self.inner_char = inner_char
        self.word_rnn_size = word_rnn_size
        self.embed_dropout_rate = embed_dropout_rate

        if self.inner_char:
            self.integration_method = integration_method
            self.char_feature_method = char_feature_method
            self.max_charlen = max_charlen
            self.nb_char_tokens = nb_char_tokens
            self.char_embedding_dim = char_embedding_dim
            if char_feature_method == 'rnn':
                if self.integration_method == 'attention':
                    self.char_rnn_size = int(self.embedding_dim / 2)
                else:
                    self.char_rnn_size = char_rnn_size
            elif char_feature_method == 'cnn':
                self.nb_filters = nb_filters
                self.conv_kernel_size = conv_kernel_size
                if self.integration_method == 'attention':
                    self.nb_filters = self.embedding_dim
        if embeddings is not None:
            self.token_embeddings = [embeddings]
        else:
            self.token_embeddings = None
        if char_feature_method == 'rnn':
            self.mask_zero = True
        else:
            self.mask_zero = False
        self.char_lstm = LSTM(char_rnn_size, return_sequences=False)
        self.char_gru = GRU(char_rnn_size, return_sequences=False)
        self.conv = Conv1D(
            kernel_size=conv_kernel_size, filters=self.nb_filters, padding='same')
        self.fc_tanh = Dense(
            embedding_dim, kernel_initializer="glorot_uniform", activation='tanh')
        self.fc_sigmoid = Dense(embedding_dim, activation='sigmoid')

        self.invalid_params = {'char_lstm', 'char_gru', 'mask_zero',
                               'conv', 'fc_tanh', 'fc_sigmoid'}

    def forward(self):
        word_ids = Input(shape=(self.maxlen,), dtype='int32', name='token')
        input_data = [word_ids]
        x = Token_Embedding(word_ids, self.nb_tokens, self.embedding_dim,
                            self.token_embeddings, True, self.maxlen,
                            self.embed_dropout_rate)

        # char features
        if self.inner_char:
            char_ids = Input(batch_shape=(None, None, None),
                             dtype='int32', name='char')
            input_data.append(char_ids)
            x_c = Token_Embedding(
                char_ids, input_dim=self.nb_char_tokens,
                output_dim=self.char_embedding_dim,
                mask_zero=self.mask_zero, name='char_embeddings',
                time_distributed=True)
            if self.char_feature_method == 'rnn':
                if self.rnn_type == 'lstm':
                    char_feature = TimeDistributed(
                        Bidirectional(self.char_lstm), name="char_lstm")(x_c)
                elif self.rnn_type == 'gru':
                    char_feature = TimeDistributed(
                        Bidirectional(self.char_gru), name="char_gru")(x_c)
                else:
                    print('invalid rnn type, only support lstm and gru')
                    sys.exit()
            elif self.char_feature_method == 'cnn':
                conv1d_out = TimeDistributed(self.conv, name='char_cnn')(x_c)
                char_feature = TimeDistributed(
                    GlobalMaxPooling1D(), name='char_pooling')(conv1d_out)
            if self.integration_method == 'concat':
                concat_tensor = concatenate([x, char_feature], axis=-1, name='concat_feature')
            elif self.integration_method == 'attention':
                word_embed_dense = self.fc_tanh(x)
                char_embed_dense = self.fc_tanh(char_feature)
                attention_evidence_tensor = add(
                    [word_embed_dense, char_embed_dense])
                attention_output = self.fc_sigmoid(attention_evidence_tensor)
                part1 = multiply([attention_output, x])
                tmp = subtract([Lambda(lambda x: K.ones_like(x))(
                    attention_output), attention_output])
                part2 = multiply([tmp, char_feature])
                concat_tensor = add([part1, part2], name='attention_feature')

        # rnn encoder
        if self.inner_char:
            enc = concat_tensor
        else:
            enc = x
        for i in range(self.nb_rnn_layers):
            if self.rnn_type == 'lstm':
                enc = Bidirectional(
                    LSTM(self.word_rnn_size, dropout=self.drop_rate,
                         recurrent_dropout=self.re_drop_rate,
                         return_sequences=True), name='word_lstm_%d' % (i+1))(enc)
            elif self.rnn_type == 'gru':
                enc = Bidirectional(
                    GRU(self.word_rnn_size, dropout=self.drop_rate,
                        recurrent_dropout=self.re_drop_rate,
                        return_sequences=True), name='word_gru_%d' % (i+1))(enc)
            else:
                print('invalid rnn type, only support lstm and gru')
                sys.exit()

        # output logits
        outputs, self._loss, self._acc = sl_output_logits(
            enc, self.nb_classes, self.use_crf)
        self.model = Model(inputs=input_data, outputs=outputs)

    def get_loss(self):
        return self._loss

    def get_metrics(self):
        return self._acc
