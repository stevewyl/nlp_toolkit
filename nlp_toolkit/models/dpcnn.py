from nlp_toolkit.models import Base_Model
from nlp_toolkit.modules.logits import tc_output_logits
from nlp_toolkit.modules.token_embedders import Token_Embedding
from keras.layers import Input, Dense, add, Activation
from keras.layers import Conv1D, MaxPooling1D, GlobalMaxPooling1D
from keras.layers.merge import concatenate
from keras.models import Model


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
                 repeat_time=2,
                 embed_dropout_rate=0.15, final_dropout_rate=0.25):
        super(DPCNN).__init__()
        self.nb_classes = nb_classes
        self.nb_tokens = nb_tokens
        self.maxlen = maxlen
        self.embedding_dim = embedding_dim
        if embeddings is not None:
            self.token_embeddings = [embeddings]
        else:
            self.token_embeddings = None
        self.region_kernel_size = region_kernel_size
        self.conv_kernel_size = conv_kernel_size
        self.nb_filters = nb_filters
        self.pool_size = pool_size
        self.repeat_time = repeat_time
        self.embed_dropout_rate = embed_dropout_rate
        self.final_dropout_rate = final_dropout_rate
        self.invalid_params = {}

    def forward(self):
        model_input = Input(shape=(self.maxlen,), dtype='int32', name='token')
        # region embedding
        x = Token_Embedding(model_input, self.nb_tokens, self.embedding_dim,
                            self.token_embeddings, False, self.maxlen,
                            self.embed_dropout_rate, name='token_embeddings')
        if isinstance(self.region_kernel_size, list):
            region = [Conv1D(self.nb_filters, f, padding='same')(x)
                      for f in self.region_kernel_size]
            region_embedding = add(region, name='region_embeddings')
        else:
            region_embedding = Conv1D(
                self.nb_filters, self.region_kernel_size, padding='same', name='region_embeddings')(x)
        # same padding convolution
        x = Activation('relu')(region_embedding)
        x = Conv1D(self.nb_filters, self.conv_kernel_size,
                   padding='same', name='conv_1')(x)
        x = Activation('relu')(x)
        x = Conv1D(self.nb_filters, self.conv_kernel_size,
                   padding='same', name='conv_2')(x)
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
