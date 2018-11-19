"""
Some custom layers, such as attention-based layers, position embedding layer
"""

from keras import initializers
from keras.engine import InputSpec, Layer
from keras import backend as K


def custom_binary_crossentropy(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_true, y_pred[:, :2]), axis=-1)


def custom_categorical_crossentropy(y_true, y_pred, n):
    return K.categorical_crossentropy(y_true, y_pred[:, :n])


class Attention(Layer):
    """
    Basic attention layer.
    Attention layers are normally used to find important tokens based on different labels.
    uses 'max trick' for numerical stability
    # Arguments:
        1. use_bias: whether to use bias
        2. use_context: whether to use context vector
        3. return_attention: whether to return attention weights as part of output
        4. attention_dim: dimensionality of the inner attention
        5. activation: whether to use activation func in first MLP
    # Inputs:
        Tensor with shape (batch_size, time_steps, hidden_size)
    # Returns:
        Tensor with shape (batch_size, hidden_size)
        If return attention weight,
        an additional tensor with shape (batch_size, time_steps) will be returned.
    """

    def __init__(self,
                 use_bias=True,
                 use_context=True,
                 return_attention=False,
                 attention_dim=None,
                 activation=True,
                 **kwargs):
        self.use_bias = use_bias
        self.use_context = use_context
        self.return_attention = return_attention
        self.attention_dim = attention_dim
        self.activation = activation
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        if len(input_shape) < 3:
            raise ValueError(
                "Expected input shape of `(batch_size, time_steps, features)`, found `{}`".format(input_shape))
        if self.attention_dim is None:
            attention_dim = input_shape[-1]
        else:
            attention_dim = self.attention_dim

        self.kernel = self.add_weight(name='kernel',
                                      shape=(input_shape[-1], attention_dim),
                                      initializer="glorot_normal",
                                      trainable=True)
        if self.use_bias:
            self.bias = self.add_weight(name='bias',
                                        shape=(attention_dim,),
                                        initializer="zeros",
                                        trainable=True)
        else:
            self.bias = None
        if self.use_context:
            self.context_kernel = self.add_weight(name='context_kernel',
                                                  shape=(attention_dim, 1),
                                                  initializer="glorot_normal",
                                                  trainable=True)
        else:
            self.context_kernel = None

        super(Attention, self).build(input_shape)

    def call(self, x, mask=None):
        # MLP
        ut = K.dot(x, self.kernel)
        if self.use_bias:
            ut = K.bias_add(ut, self.bias)
        if self.activation:
            ut = K.tanh(ut)
        if self.context_kernel:
            ut = K.dot(ut, self.context_kernel)
        ut = K.squeeze(ut, axis=-1)
        # softmax
        at = K.exp(ut - K.max(ut, axis=-1, keepdims=True))
        if mask is not None:
            at *= K.cast(mask, K.floatx())
        att_weights = at / (K.sum(at, axis=1, keepdims=True) + K.epsilon())
        # output
        atx = x * K.expand_dims(att_weights, axis=-1)
        output = K.sum(atx, axis=1)
        if self.return_attention:
            return [output, att_weights]
        return output

    def compute_mask(self, input, input_mask=None):
        if isinstance(input_mask, list):
            return [None] * len(input_mask)
        else:
            return None

    def compute_output_shape(self, input_shape):
        output_len = input_shape[2]
        if self.return_attention:
            return [(input_shape[0], output_len), (input_shape[0], input_shape[1])]
        return (input_shape[0], output_len)


class Self_Attentive(Layer):
    """
    From "A Structured Self-Attentive Sentence Embedding" (2017)
    """

    def __init__(self, ws1, ws2, punish, init='glorot_normal', **kwargs):
        self.kernel_initializer = initializers.get(init)
        self.weight_ws1 = ws1
        self.weight_ws2 = ws2
        self.punish = punish
        super(Self_Attentive, self).__init__(** kwargs)

    def build(self, input_shape):
        self.Ws1 = self.add_weight(shape=(input_shape[-1], self.weight_ws1),
                                   initializer=self.kernel_initializer,
                                   trainable=True,
                                   name='{}_Ws1'.format(self.name))
        self.Ws2 = self.add_weight(shape=(self.weight_ws1, self.weight_ws2),
                                   initializer=self.kernel_initializer,
                                   trainable=True,
                                   name='{}_Ws2'.format(self.name))
        self.batch_size = input_shape[0]
        super(Self_Attentive, self).build(input_shape)

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        uit = K.tanh(K.dot(x, self.Ws1))
        ait = K.dot(uit, self.Ws2)
        ait = K.permute_dimensions(ait, (0, 2, 1))
        A = K.softmax(ait, axis=1)
        M = K.batch_dot(A, x)
        if self.punish:
            A_T = K.permute_dimensions(A, (0, 2, 1))
            tile_eye = K.tile(K.eye(self.weight_ws2), [self.batch_size, 1])
            tile_eye = K.reshape(
                tile_eye, shape=[-1, self.weight_ws2, self.weight_ws2])
            AA_T = K.batch_dot(A, A_T) - tile_eye
            P = K.l2_normalize(AA_T, axis=(1, 2))
            return M, P
        else:
            return M

    def compute_output_shape(self, input_shape):
        if self.punish:
            out1 = (input_shape[0], self.weight_ws2, input_shape[-1])
            out2 = (input_shape[0], self.weight_ws2, self.weight_ws2)
            return [out1, out2]
        else:
            return (input_shape[0], self.weight_ws2, input_shape[-1])


class Position_Embedding(Layer):
    """
    Computes sequence position information for Attention based models
    https://github.com/bojone/attention/blob/master/attention_keras.py

    # Arguments:
        A tensor with shape (batch_size, seq_len, word_size)
    # Returns:
        A position tensor with shape (batch_size, seq_len, position_size)
    """

    def __init__(self, size=None, mode='sum', **kwargs):
        self.size = size  # 必须为偶数
        self.mode = mode
        super(Position_Embedding, self).__init__(**kwargs)

    def call(self, x):
        if (self.size is None) or (self.mode == 'sum'):
            self.size = int(x.shape[-1])
        batch_size, seq_len = K.shape(x)[0], K.shape(x)[1]
        position_j = 1. / K.pow(10000.,
                                2 * K.arange(self.size / 2, dtype='float32'
                                             ) / self.size)
        position_j = K.expand_dims(position_j, 0)
        # K.arange不支持变长，只好用这种方法生成
        position_i = K.cumsum(K.ones_like(x[:, :, 0]), 1) - 1
        position_i = K.expand_dims(position_i, 2)
        position_ij = K.dot(position_i, position_j)
        position_ij = K.concatenate(
            [K.cos(position_ij), K.sin(position_ij)], 2)
        if self.mode == 'sum':
            return position_ij + x
        elif self.mode == 'concat':
            return K.concatenate([position_ij, x], 2)

    def compute_output_shape(self, input_shape):
        if self.mode == 'sum':
            return input_shape
        elif self.mode == 'concat':
            return (input_shape[0], input_shape[1], input_shape[2] + self.size)


class Multi_Head_Attention(Layer):
    """
    Multi_Head Attention Layer defined in <Attention is all your need>.
    If you want to use it as self-attention, then pass in three same tensors
    https://github.com/bojone/attention/blob/master/attention_keras.py
    """

    def __init__(self, nb_head, size_per_head, **kwargs):
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.output_dim = nb_head*size_per_head
        super(Multi_Head_Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        self.WQ = self.add_weight(name='WQ',
                                  shape=(input_shape[0][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WK = self.add_weight(name='WK',
                                  shape=(input_shape[1][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        self.WV = self.add_weight(name='WV',
                                  shape=(input_shape[2][-1], self.output_dim),
                                  initializer='glorot_uniform',
                                  trainable=True)
        super(Multi_Head_Attention, self).build(input_shape)

    def Mask(self, inputs, seq_len, mode='mul'):
        """
        # Arguments:
            inputs: input tensor with shape (batch_size, seq_len, input_size)
            seq_len: Each sequence's actual length with shape (batch_size,)
            mode:
                mul: mask the rest dim with zero, used before fully-connected layer
                add: subtract a big constant from the rest, used before softmax layer
        # Reutrns:
            Masked tensors with the same shape of input tensor
        """
        if seq_len is None:
            return inputs
        else:
            mask = K.one_hot(seq_len[:, 0], K.shape(inputs)[1])
            mask = 1 - K.cumsum(mask, 1)
            for _ in range(len(inputs.shape) - 2):
                mask = K.expand_dims(mask, 2)
            if mode == 'mul':
                return inputs * mask
            if mode == 'add':
                return inputs - (1 - mask) * 1e12

    def call(self, x):
        # if only pass in [Q_seq,K_seq,V_seq], then no Mask operation
        # if you also pass in [Q_len,V_len], Mask will apply to the redundance
        if len(x) == 3:
            Q_seq, K_seq, V_seq = x
            Q_len, V_len = None, None
        elif len(x) == 5:
            Q_seq, K_seq, V_seq, Q_len, V_len = x
        # linear transformation of Q, K, V
        Q_seq = K.dot(Q_seq, self.WQ)
        Q_seq = K.reshape(
            Q_seq, (-1, K.shape(Q_seq)[1], self.nb_head, self.size_per_head))
        Q_seq = K.permute_dimensions(Q_seq, (0, 2, 1, 3))
        K_seq = K.dot(K_seq, self.WK)
        K_seq = K.reshape(
            K_seq, (-1, K.shape(K_seq)[1], self.nb_head, self.size_per_head))
        K_seq = K.permute_dimensions(K_seq, (0, 2, 1, 3))
        V_seq = K.dot(V_seq, self.WV)
        V_seq = K.reshape(
            V_seq, (-1, K.shape(V_seq)[1], self.nb_head, self.size_per_head))
        V_seq = K.permute_dimensions(V_seq, (0, 2, 1, 3))
        # compute inner product, then mask, then softmax
        A = K.batch_dot(Q_seq, K_seq, axes=[3, 3]) / self.size_per_head ** 0.5
        A = K.permute_dimensions(A, (0, 3, 2, 1))
        A = self.Mask(A, V_len, 'add')
        A = K.permute_dimensions(A, (0, 3, 2, 1))
        A = K.softmax(A)
        # output and mask
        O_seq = K.batch_dot(A, V_seq, axes=[3, 2])
        O_seq = K.permute_dimensions(O_seq, (0, 2, 1, 3))
        O_seq = K.reshape(O_seq, (-1, K.shape(O_seq)[1], self.output_dim))
        O_seq = self.Mask(O_seq, Q_len, 'mul')
        return O_seq

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0], input_shape[0][1], self.output_dim)


def kmax_pooling(Layer):
    pass


# TODO to handle all kinds of embedding layer variants 
def feature_embedding(features, inner_char=False):
    pass
