"""
Model Zoos.
TODO add shape size for each layer
"""

import json
import numpy as np
import tensorflow as tf
import keras.backend as K
from keras.models import Model, Sequential
from keras.regularizers import L1L2, l2
from keras.layers.merge import concatenate
from keras.layers import Input, Embedding, Dense
from keras.layers import SpatialDropout1D, Dropout
from keras.layers import multiply, add, subtract
from keras.layers import LSTM, GRU, Bidirectional, Conv1D
from keras.layers import TimeDistributed, Activation, Flatten, Lambda
from keras.layers import GlobalMaxPooling1D, GlobalAveragePooling1D, MaxPooling1D
from keras.layers import BatchNormalization
from keras_contrib.layers import CRF
from nlp_toolkit.layer import Attention, Multi_Head_Attention, Position_Embedding
from nlp_toolkit.layer import custom_binary_crossentropy, custom_categorical_crossentropy
from nlp_toolkit.utilities import logger


class Base_Model(object):
    def __init__(self):
        self.model = None

    def save(self, weights_file, params_file):
        self.save_weights(weights_file)
        self.save_params(params_file)

    def save_params(self, file_path):
        with open(file_path, 'w') as f:
            invalid_params = {'_loss', '_acc', 'model',
                              'embeddings', 'char_embeddings', 'token_embeddings',
                              'region_embeddings', 'word_embeddings',
                              'word_embed', 'embed', 'embed_drop', 'char_embed',
                              'activation', 'lstm', 'pool', 'attention', 'attention_layer',
                              'cnn_list', 'pool_list', 'conv',
                              'char_lstm', 'char_gru', 'fc_sigmoid', 'fc_tanh',
                              'word_lstm', 'word_gru', 'fc',
                              'pos_embed_layer', 'transformers'}
            params = {name.lstrip('_'): val for name, val in vars(self).items()
                      if name not in invalid_params}
            print('model hyperparameters:\n', params)
            json.dump(params, f, sort_keys=True, indent=4)

    def save_weights(self, filepath):
        self.model.save_weights(filepath)

    @classmethod
    def load(cls, weights_file, params_file):
        params = cls.load_params(params_file)
        self = cls(**params)
        self.forward()
        self.model.load_weights(weights_file)
        print('model loaded')
        return self

    @classmethod
    def load_params(cls, file_path):
        with open(file_path) as f:
            params = json.load(f)
        return params


def tc_output_logits(x, nb_classes, final_dropout_rate=0):
    if final_dropout_rate != 0:
        x = Dropout(final_dropout_rate)(x)
    if nb_classes > 2:
        activation_func = 'softmax'
    else:
        activation_func = 'sigmoid'
    logits = Dense(nb_classes, kernel_regularizer=l2(0.01),
                   activation=activation_func, name='softmax')(x)
    outputs = [logits]
    return outputs


def sl_output_logits(x, nb_classes, use_crf=True):
    if use_crf:
        crf = CRF(nb_classes, sparse_target=False)
        loss = crf.loss_function
        acc = [crf.accuracy]
        outputs = crf(x)
    else:
        loss = 'categorical_crossentropy'
        acc = ['acc']
        outputs = Dense(nb_classes, activation='softmax')(x)
    return outputs, loss, acc


class textCNN(Base_Model):
    """
    The known Kim CNN model used in text classification.
    It use mulit-channel CNN to encode texts
    """

    def __init__(self, nb_classes, nb_tokens, maxlen,
                 embedding_dim=256, embeddings=None, embed_l2=1E-6,
                 conv_kernel_size=[3, 4, 5], pool_size=[2, 2, 2],
                 nb_filters=128, fc_size=128,
                 final_dropout_rate=0):
        super(textCNN).__init__()
        self.nb_classes = nb_classes
        self.nb_tokens = nb_tokens
        self.maxlen = maxlen
        self.embedding_dim = embedding_dim
        self.nb_filters = nb_filters
        self.pool_size = pool_size
        self.conv_kernel_size = conv_kernel_size
        self.fc_size = fc_size
        self.final_dropout_rate = final_dropout_rate

        # core layer: multi-channel cnn-pool layers
        self.cnn_list = [Conv1D(
            nb_filters, f, padding='same', name='conv_%d' % k) for k, f in enumerate(conv_kernel_size)]
        self.pool_list = [MaxPooling1D(p, name='pool_%d' % k) for k, p in enumerate(pool_size)]
        self.fc = Dense(fc_size, activation='relu', kernel_initializer='he_normal')

        # embedding layer
        if embeddings is not None:
            word_embeddings = [embeddings]
        else:
            word_embeddings = embeddings
        embed_reg = L1L2(l2=embed_l2) if embed_l2 != 0 else None
        self.embed = Embedding(
            input_dim=nb_tokens,
            output_dim=embedding_dim,
            weights=word_embeddings,
            input_length=maxlen,
            embeddings_regularizer=embed_reg,
            name='token_embeddings')

    def forward(self):
        model_input = Input(shape=(self.maxlen,), dtype='int32', name='token')
        x = self.embed(model_input)
        cnn_combine = []

        for i in range(len(self.conv_kernel_size)):
            cnn = self.cnn_list[i](x)
            pool = self.pool_list[i](cnn)
            cnn_combine.append(pool)
        x = concatenate(cnn_combine, axis=-1)

        x = Flatten()(x)
        x = Dropout(self.final_dropout_rate)(x)
        x = self.fc(x)

        outputs = tc_output_logits(x, self.nb_classes, self.final_dropout_rate)

        self.model = Model(inputs=model_input,
                           outputs=outputs, name="TextCNN")

    def get_loss(self):
        if self.nb_classes == 2:
            return 'binary_crossentropy'
        elif self.nb_classes > 2:
            return 'categorical_crossentropy'

    def get_metrics(self):
        return ['acc']


class bi_lstm_attention(Base_Model):
    """
    Model from DeepMoji.

    Model structure: double bi-lstm followed by attention with some dropout techniques

    # Arguments:
        nb_classes: nbber of classes in the dataset.
        nb_tokens: nbber of tokens in the dataset (i.e. vocabulary size).
        maxlen: Maximum length of a token.
        embedding_dim: Embedding layer output dim.
        embeddings: Embedding weights. Default word embeddings.
        feature_output: If True the model returns the penultimate
                        feature vector rather than Softmax probabilities
                        (defaults to False).
        embed_dropout_rate: Dropout rate for the embedding layer.
        final_dropout_rate: Dropout rate for the final Softmax layer.
        embed_l2: L2 regularization for the embedding layerl.

    # Returns:
        Model with the given parameters.
    """

    def __init__(self, nb_classes, nb_tokens, maxlen,
                 embedding_dim=256, embeddings=None,
                 rnn_size=512, attention_dim=None,
                 embed_dropout_rate=0,
                 final_dropout_rate=0, embed_l2=1E-6,
                 return_attention=False):
        super(bi_lstm_attention).__init__()
        self.nb_classes = nb_classes
        self.nb_tokens = nb_tokens
        self.maxlen = maxlen
        self.embedding_dim = embedding_dim
        self.rnn_size = rnn_size
        self.attention_dim = attention_dim
        if embeddings is not None:
            word_embeddings = [embeddings]
        else:
            word_embeddings = embeddings
        self.embed_dropout_rate = embed_dropout_rate
        self.final_dropout_rate = final_dropout_rate
        self.return_attention = return_attention
        # embedding layer
        embed_reg = L1L2(l2=embed_l2) if embed_l2 != 0 else None
        self.embed = Embedding(
            input_dim=nb_tokens,
            output_dim=embedding_dim,
            weights=word_embeddings,
            mask_zero=True,
            input_length=maxlen,
            embeddings_regularizer=embed_reg,
            name='token_embeddings')

        self.activation = Activation('tanh')
        self.embed_drop = SpatialDropout1D(
            embed_dropout_rate, name='embed_drop')

        # core layer：attention
        self.attention_layer = Attention(
            attention_dim=attention_dim,
            return_attention=return_attention, name='attlayer')

    def forward(self):
        model_input = Input(shape=(self.maxlen,), dtype='int32', name='token')
        x = self.embed(model_input)
        x = self.activation(x)

        # entire embedding channels are dropped out instead of the
        # normal Keras embedding dropout, which drops all channels for entire words
        # many of the datasets contain so few words that losing one or more words can alter the emotions completely
        if self.embed_dropout_rate != 0:
            x = self.embed_drop(x)

        # skip-connection from embedding to output eases gradient-flow and allows access to lower-level features
        # ordering of the way the merge is done is important for consistency with the pretrained model
        lstm_0_output = Bidirectional(
            LSTM(self.rnn_size, return_sequences=True), name="bi_lstm_0")(x)
        lstm_1_output = Bidirectional(
            LSTM(self.rnn_size, return_sequences=True), name="bi_lstm_1")(lstm_0_output)
        x = concatenate([lstm_1_output, lstm_0_output, x], name='concatenate')

        x = self.attention_layer(x)
        if self.return_attention:
            x, weights = x
        outputs = tc_output_logits(x, self.nb_classes, self.final_dropout_rate)
        if self.return_attention:
            outputs.append(weights)
            outputs = concatenate(outputs, axis=-1, name='outputs')

        self.model = Model(inputs=model_input,
                           outputs=outputs, name="Bi_LSTM_Attention")

    def get_loss(self):
        if self.nb_classes == 2:
            if self.return_attention:
                return custom_binary_crossentropy
            else:
                return 'binary_crossentropy'
        elif self.nb_classes > 2:
            if self.return_attention:
                return custom_categorical_crossentropy
            else:
                return 'categorical_crossentropy'

    def get_metrics(self):
        return ['acc']


class multi_head_self_attention(Base_Model):
    """
    Multi-Head Self Attention Model.
    Use Transfomer's architecture to encode texts.

    # Arguments:
        1. nb_transformer: the nbber of attention layer.
        2. nb_head: the nbber of attention block in one layer
        3. head_size: the hidden size of each attention unit
        4. pos_embed: whether to use poisition embedding
    """

    def __init__(self, nb_classes, nb_tokens, maxlen,
                 nb_head=8, head_size=16, nb_transfomer=2,
                 embedding_dim=256, embeddings=None, embed_l2=1E-6,
                 pos_embed=False, final_dropout_rate=0):
        self.nb_classes = nb_classes
        self.nb_tokens = nb_tokens
        self.maxlen = maxlen
        self.nb_head = nb_head
        self.head_size = head_size
        self.embedding_dim = embedding_dim
        self.nb_transfomer = nb_transfomer
        if embeddings is not None:
            word_embeddings = [embeddings]
        else:
            word_embeddings = embeddings
        self.pos_embed = pos_embed
        self.final_dropout_rate = final_dropout_rate

        embed_reg = L1L2(l2=embed_l2) if embed_l2 != 0 else None
        self.embed = Embedding(
            input_dim=nb_tokens,
            output_dim=embedding_dim,
            weights=word_embeddings,
            input_length=maxlen,
            embeddings_regularizer=embed_reg,
            name='token_embeddings')
        self.pos_embed_layer = Position_Embedding(name='position_embedding')
        self.transformers = [Multi_Head_Attention(
            nb_head, head_size, name='self_attention_%d' % i) for i in range(nb_transfomer)]
        self.pool = GlobalAveragePooling1D()

    def forward(self):
        model_input = Input(shape=(self.maxlen,), dtype='int32', name='token')
        x = self.embed(model_input)
        if self.pos_embed:
            x = self.pos_embed_layer(x)
        for i in range(self.nb_transfomer):
            x = self.transformers[i]([x, x, x])
        x = self.pool(x)
        outputs = tc_output_logits(x, self.nb_classes, self.final_dropout_rate)
        self.model = Model(inputs=model_input,
                           outputs=outputs, name="Self_Multi_Head_Attention")

    def get_loss(self):
        if self.nb_classes == 2:
            return 'binary_crossentropy'
        elif self.nb_classes > 2:
            return 'categorical_crossentropy'

    def get_metrics(self):
        return ['acc']


class DPCNN(Base_Model):
    """
    Deep Pyramid CNN
    Three key point of DPCNN:
        1. region embeddings
        2. fixed feature maps
        3. residual connection
    """

    def __init__(self, nb_classes, nb_tokens, maxlen,
                 embedding_dim=256, embeddings=None,
                 region_kernel_size=[3, 4, 5],
                 conv_kernel_size=3, nb_filters=250, pool_size=3,
                 repeat_time=2, final_dropout_rate=0.25):
        super(DPCNN).__init__()
        self.nb_classes = nb_classes
        self.nb_tokens = nb_tokens
        self.maxlen = maxlen
        self.embedding_dim = embedding_dim
        if embeddings is not None:
            self.word_embeddings = [embeddings]
        else:
            self.word_embeddings = None
        self.region_kernel_size = region_kernel_size
        self.conv_kernel_size = conv_kernel_size
        self.nb_filters = nb_filters
        self.pool_size = pool_size
        self.repeat_time = repeat_time
        self.final_dropout_rate = final_dropout_rate

    def forward(self):
        model_input = Input(shape=(self.maxlen,), dtype='int32', name='token')
        # region embedding
        x = Embedding(self.nb_tokens, self.embedding_dim, input_length=self.maxlen,
                      weights=self.word_embeddings, name='token_embeddings')(model_input)
        if isinstance(self.region_kernel_size, list):
            region = [Conv1D(self.nb_filters, f, padding='same')(x) for f in self.region_kernel_size]
            region_embedding = add(region, name='region_embeddings')
        else:
            region_embedding = Conv1D(
                self.nb_filters, self.region_kernel_size, padding='same', name='region_embeddings')(x)
        # same padding convolution
        x = Activation('relu')(region_embedding)
        x = Conv1D(self.nb_filters, self.conv_kernel_size, padding='same', name='conv_1')(x)
        x = Activation('relu')(x)
        x = Conv1D(self.nb_filters, self.conv_kernel_size, padding='same', name='conv_2')(x)
        # residual connection
        x = add([x, region_embedding], name='pre_block_hidden')

        for k in range(self.repeat_time):
            x = self._block(x, k)
        x = GlobalMaxPooling1D()(x)
        outputs = tc_output_logits(x, self.nb_classes, self.final_dropout_rate)

        self.model = Model(inputs=model_input,
                           outputs=outputs, name="Deep Pyramid CNN")

    def _block(self, x, k):
        x = MaxPooling1D(self.pool_size, strides=2)(x)
        last_x = x
        x = Activation('relu')(x)
        x = Conv1D(self.nb_filters, self.conv_kernel_size,
                   padding='same', name='block_%d_conv_1' % k)(x)
        x = Activation('relu')(x)
        x = Conv1D(self.nb_filters, self.conv_kernel_size,
                   padding='same', name='block_%d_conv_2' % k)(x)
        # residual connection
        x = add([x, last_x])
        return x

    def get_loss(self):
        if self.nb_classes == 2:
            return 'binary_crossentropy'
        elif self.nb_classes > 2:
            return 'categorical_crossentropy'

    def get_metrics(self):
        return ['acc']


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
                 embed_l2=1E-6):
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
            word_embeddings = [embeddings]
        else:
            word_embeddings = embeddings
        if char_feature_method == 'rnn':
            mask_zero = True
        else:
            mask_zero = False

        embed_reg = L1L2(l2=embed_l2) if embed_l2 != 0 else None
        self.word_embed = Embedding(
            input_dim=nb_tokens,
            output_dim=embedding_dim,
            weights=word_embeddings,
            mask_zero=mask_zero,
            embeddings_regularizer=embed_reg,
            name='word_embedding')

        self.char_embed = TimeDistributed(
            Embedding(
                input_dim=nb_char_tokens, output_dim=char_embedding_dim,
                mask_zero=mask_zero, name='char_embedding'))

        self.char_lstm = LSTM(char_rnn_size, return_sequences=False)
        self.char_gru = GRU(char_rnn_size, return_sequences=False)
        self.conv = Conv1D(
            kernel_size=conv_kernel_size, filters=nb_filters, padding='same')
        self.fc_tanh = Dense(
            embedding_dim, kernel_initializer="glorot_uniform", activation='tanh')
        self.fc_sigmoid = Dense(embedding_dim, activation='sigmoid')

    def forward(self):
        word_ids = Input(shape=(self.maxlen,), dtype='int32', name='token')
        input_data = [word_ids]
        x = self.word_embed(word_ids)

        # char features
        if self.inner_char:
            char_ids = Input(batch_shape=(None, None, None),
                             dtype='int32', name='char')
            input_data.append(char_ids)
            x_c = self.char_embed(char_ids)
            if self.char_feature_method == 'rnn':
                if self.rnn_type == 'lstm':
                    char_feature = TimeDistributed(
                        Bidirectional(self.char_lstm), name="char_lstm")(x_c)
                elif self.rnn_type == 'gru':
                    char_feature = TimeDistributed(
                        Bidirectional(self.char_gru), name="char_gru")(x_c)
                else:
                    logger.warning('invalid rnn type, only support lstm and gru')
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
                logger.warning('invalid rnn type, only support lstm and gru')
                break

        # output logits
        outputs, self._loss, self._acc = sl_output_logits(
            enc, self.nb_classes, self.use_crf)
        self.model = Model(inputs=input_data, outputs=outputs)

    def get_loss(self):
        return self._loss

    def get_metrics(self):
        return self._acc


class Char_RNN(Base_Model):
    """
    Similar model structure to Word_RNN. But use char as basic token.
    And some useful features are included: 1. radicals 2. segmentation tag 3. nchar
    """

    def __init__(self, nb_classes, nb_tokens, maxlen,
                 emebdding_dim=64, use_crf=True,
                 use_seg=False, use_radical=False,
                 nb_seg_tokens=None, nb_radical_tokens=None,
                 rnn_type='lstm', nb_rnn_layers=2,
                 char_rnn_size=128, drop_rate=0.5,
                 re_drop_rate=0.15):
        self.nb_classes = nb_classes
        self.nb_tokens = nb_tokens
        self.maxlen = maxlen
        self.embedding_dim = emebdding_dim
        self.use_crf = use_crf
        self.use_seg = use_seg
        self.use_radical = use_radical
        self.rnn_type = rnn_type
        self.nb_rnn_layers = nb_rnn_layers
        self.drop_rate = drop_rate
        self.re_drop_rate = re_drop_rate
        self.char_rnn_size = char_rnn_size
        if use_seg:
            self.nb_seg_tokens = nb_seg_tokens
        if use_radical:
            self.nb_radical_tokens = nb_radical_tokens

        super(Char_RNN).__init__()

    def forward(self):
        char_ids = Input(shape=(self.maxlen,), dtype='int32', name='token')
        input_data = [char_ids]
        char_embed = Embedding(input_dim=self.nb_tokens,
                               output_dim=self.embedding_dim,
                               mask_zero=True,
                               name='char_embedding')(char_ids)
        embed_features = [char_embed]
        if self.use_seg:
            seg_ids = Input(shape=(self.maxlen,), dtype='int32', name='seg')
            input_data.append(seg_ids)
            seg_emebd = Embedding(input_dim=self.nb_seg_tokens,
                                  output_dim=8, mask_zero=True,
                                  name='seg_embedding')(seg_ids)
            embed_features.append(seg_emebd)
        if self.use_radical:
            radical_ids = Input(shape=(self.maxlen,), dtype='int32', name='radical')
            input_data.append(radical_ids)
            radical_embed = Embedding(input_dim=self.nb_radical_tokens,
                                      output_dim=32, mask_zero=True,
                                      name='radical_embedding')(radical_ids)
            embed_features.append(radical_embed)
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
                logger.warning('invalid rnn type, only support lstm and gru')
                break

        outputs, self._loss, self._acc = sl_output_logits(
            x, self.nb_classes, self.use_crf)
        self.model = Model(inputs=input_data, outputs=outputs)

    def get_loss(self):
        return self._loss

    def get_metrics(self):
        return self._acc


class IDCNN(Base_Model):
    """
    Iterated Dilated Convolution Nerual Networks with CRF
    """

    def __init__(self, nb_classes,
                 nb_tokens,
                 maxlen,
                 embeddings=None,
                 embedding_dim=64,
                 drop_rate=0.25,
                 nb_filters=64,
                 conv_kernel_size=3,
                 dilation_rate=[1, 1, 2],
                 repeat_times=4,
                 use_crf=True,
                 ):
        super(IDCNN).__init__()
        self.nb_classes = nb_classes
        self.nb_tokens = nb_tokens
        self.maxlen = maxlen
        self.embedding_dim = embedding_dim
        self.drop_rate = drop_rate
        self.nb_filters = nb_filters
        self.conv_kernel_size = conv_kernel_size
        self.dilation_rate = dilation_rate
        self.repeat_times = repeat_times
        self.use_crf = use_crf
        if embeddings is not None:
            self.word_embeddings = [embeddings]
        else:
            self.word_embeddings = embeddings

    def forward(self):
        word_ids = Input(shape=(self.maxlen,), dtype='int32', name='token')
        input_data = [word_ids]
        embed = Embedding(input_dim=self.nb_tokens,
                          output_dim=self.embedding_dim,
                          weights=self.word_embeddings,
                          name='word_embed')(word_ids)
        if self.drop_rate < 1.0:
            embed = Dropout(self.drop_rate)(embed)
        layerInput = Conv1D(
            self.nb_filters, self.conv_kernel_size, padding='same', name='conv_first')(embed)
        dilation_layers = []
        totalWidthForLastDim = 0
        for j in range(self.repeat_times):
            for i in range(len(self.dilation_rate)):
                islast = True if i == len(self.dilation_rate) - 1 else False
                conv = Conv1D(self.nb_filters, self.conv_kernel_size, use_bias=True,
                              padding='same', dilation_rate=self.dilation_rate[i],
                              name='atrous_conv_%d_%d' % (j, i))(layerInput)
                conv = Activation('relu')(conv)
                if islast:
                    dilation_layers.append(conv)
                    totalWidthForLastDim += self.nb_filters
                layerInput = conv
        dilation_conv = concatenate(
            dilation_layers, axis=-1, name='dilated_conv')
        if self.drop_rate < 1.0:
            enc = Dropout(self.drop_rate)(dilation_conv)

        outputs, self._loss, self._acc = sl_output_logits(
            enc, self.nb_classes, self.use_crf)
        self.model = Model(inputs=input_data, outputs=outputs)

    def get_loss(self):
        return self._loss

    def get_metrics(self):
        return self._acc
