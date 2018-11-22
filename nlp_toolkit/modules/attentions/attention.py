from keras.engine import Layer
from keras import backend as K


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
