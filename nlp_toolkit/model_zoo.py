"""
Model Zoos.
"""

import json
import numpy as np
import tensorflow as tf
from keras.models import Model, Sequential
from keras.regularizers import L1L2, l2
from keras.layers.merge import concatenate
from keras.layers import Input, Bidirectional, Embedding, Dense, Dropout, Lambda
from keras.layers import SpatialDropout1D, LSTM, Activation, GlobalAveragePooling1D
from nlp_toolkit.layer import Attention, Multi_Head_Attention, Position_Embedding


class Base_Model(object):
    def __init__(self):
        self.model = None

    def save(self, weights_file, params_file):
        self.save_weights(weights_file)
        self.save_params(params_file)

    def save_params(self, file_path):
        with open(file_path, 'w') as f:
            invalid_params = {'_loss', 'model', 'word_embeddings', 'embeddings',
                              'char_embeddings', '_acc', 'word_embed'}
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
        self.build()
        self.model.load_weights(weights_file)
        print('model loaded')
        return self

    @classmethod
    def load_params(cls, file_path):
        with open(file_path) as f:
            params = json.load(f)
        return params    


def output_logits(x, nb_classes, final_dropout_rate=0,
                  feature_output=False):
    if not feature_output:
        # output class probabilities
        if final_dropout_rate != 0:
            x = Dropout(final_dropout_rate)(x)
        if nb_classes > 2:
            activation_func = 'softmax'
        else:
            activation_func = 'sigmoid'
        logits = Dense(nb_classes, kernel_regularizer=l2(0.01),
                       activation=activation_func, name='softmax')(x)
        outputs = [logits]
    else:
        # output penultimate feature vector
        outputs = [x]
    return outputs


class bi_lstm_attention(Base_Model):
    """
    Model from DeepMoji.

    Model structure: double bi-lstm followed by attention with some dropout techniques

    # Arguments:
        nb_classes: Number of classes in the dataset.
        nb_tokens: Number of tokens in the dataset (i.e. vocabulary size).
        maxlen: Maximum length of a token.
        embed_dim: Embedding layer output dim.
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
                 embed_dim=256, embeddings=None,
                 feature_output=False, embed_dropout_rate=0,
                 final_dropout_rate=0, embed_l2=1E-6,
                 return_attention=False):
        super(bi_lstm_attention).__init__()
        self.nb_classes = nb_classes
        self.nb_tokens = nb_tokens
        self.maxlen = maxlen
        self.embed_dim = embed_dim
        if embeddings is not None:
            self.word_embeddings = [embeddings]
        else:
            self.word_embeddings = embeddings
        self.feature_output = feature_output
        self.embed_dropout_rate = embed_dropout_rate
        self.final_dropout_rate = final_dropout_rate
        self.embed_l2 = embed_l2
        self.return_attention = return_attention

    def build(self):
        model_input = Input(shape=(self.maxlen,), dtype='int32')
        embed_reg = L1L2(l2=self.embed_l2) if self.embed_l2 != 0 else None
        embed = Embedding(input_dim=self.nb_tokens,
                          output_dim=self.embed_dim,
                          weights=self.word_embeddings,
                          mask_zero=True,
                          input_length=self.maxlen,
                          embeddings_regularizer=embed_reg,
                          name='embedding')
        x = embed(model_input)
        x = Activation('tanh')(x)

        # entire embedding channels are dropped out instead of the
        # normal Keras embedding dropout, which drops all channels for entire words
        # many of the datasets contain so few words that losing one or more words can alter the emotions completely
        if self.embed_dropout_rate != 0:
            embed_drop = SpatialDropout1D(self.embed_dropout_rate, name='embed_drop')
            x = embed_drop(x)

        # skip-connection from embedding to output eases gradient-flow and allows access to lower-level features
        # ordering of the way the merge is done is important for consistency with the pretrained model
        lstm_0_output = Bidirectional(
            LSTM(512, return_sequences=True), name="bi_lstm_0")(x)
        lstm_1_output = Bidirectional(
            LSTM(512, return_sequences=True), name="bi_lstm_1")(lstm_0_output)
        x = concatenate([lstm_1_output, lstm_0_output, x], name='concatenate')

        # normal attention layer for find important words based on different labels
        x = Attention(return_attention=self.return_attention,
                      name='attlayer')(x)
        if self.return_attention:
            x, weights = x
        outputs = output_logits(x, self.nb_classes,
                                self.final_dropout_rate, self.feature_output)
        if self.return_attention:
            outputs.append(weights)
            outputs = concatenate(outputs, axis=-1, name='outputs')

        self.model = Model(inputs=[model_input],
                           outputs=outputs, name="Bi_LSTM_Attention")


class multi_head_self_attention(Base_Model):
    """
    Multi-Head Self Attention Model
    """

    def __init__(self, nb_classes, nb_tokens, maxlen,
                 nb_head=8, head_size=16,
                 embed_dim=256, embeddings=None, embed_l2=1E-6,
                 pos_embed=False, final_dropout_rate=0):
        self.nb_classes = nb_classes
        self.nb_tokens = nb_tokens
        self.maxlen = maxlen
        self.nb_head = nb_head
        self.head_size = head_size
        self.embed_dim = embed_dim
        if embeddings is not None:
            self.word_embeddings = [embeddings]
        else:
            self.word_embeddings = embeddings
        self.embed_l2 = embed_l2
        self.pos_embed = pos_embed
        self.final_dropout_rate = final_dropout_rate

    def build(self):
        model_input = Input(shape=(self.maxlen,), dtype='int32')
        embed_reg = L1L2(l2=self.embed_l2) if self.embed_l2 != 0 else None
        embed = Embedding(input_dim=self.nb_tokens,
                          output_dim=self.embed_dim,
                          weights=self.word_embeddings,
                          input_length=self.maxlen,
                          embeddings_regularizer=embed_reg,
                          name='embedding')
        x = embed(model_input)
        if self.pos_embed:
            x = Position_Embedding(name='position_embedding')(x)
        x = Multi_Head_Attention(self.nb_head, self.head_size,
                                 name='self_attention_1')([x, x, x])
        x = Multi_Head_Attention(self.nb_head, self.head_size,
                                 name='self_attention_2')([x, x, x])
        x = GlobalAveragePooling1D()(x)
        outputs = output_logits(x, self.nb_classes, self.final_dropout_rate)
        self.model = Model(inputs=[model_input],
                           outputs=outputs, name="Self_Multi_Head_Attention")
