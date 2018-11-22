from keras.engine import Layer
from keras import backend as K
from keras import initializers


class Multi_Dim_Attention(Layer):
    """
    2D attention from "A Structured Self-Attentive Sentence Embedding" (2017)
    """

    def __init__(self, ws1, ws2, punish, init='glorot_normal', **kwargs):
        self.kernel_initializer = initializers.get(init)
        self.weight_ws1 = ws1
        self.weight_ws2 = ws2
        self.punish = punish
        super(Multi_Dim_Attention, self).__init__(** kwargs)

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
        super(Multi_Dim_Attention, self).build(input_shape)

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
