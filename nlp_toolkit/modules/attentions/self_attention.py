from keras.engine import Layer
from keras import backend as K


class Self_Attention(Layer):
    """
    Multi_Head Attention Layer defined in <Attention is all your need>.
    If you want to use it as self-attention, then pass in three same tensors
    https://github.com/bojone/attention/blob/master/attention_keras.py
    """

    def __init__(self, nb_head, size_per_head, **kwargs):
        self.nb_head = nb_head
        self.size_per_head = size_per_head
        self.output_dim = nb_head*size_per_head
        super(Self_Attention, self).__init__(**kwargs)

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
        super(Self_Attention, self).build(input_shape)

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
