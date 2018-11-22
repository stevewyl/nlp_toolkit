from nlp_toolkit.models import Base_Model
from nlp_toolkit.modules.token_embedders import Token_Embedding
from nlp_toolkit.modules.logits import sl_output_logits
from keras.layers import Input, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D
from keras.layers.merge import concatenate
from keras.models import Model


class IDCNN(Base_Model):
    """
    Iterated Dilated Convolution Nerual Networks with CRF
    """

    def __init__(self, nb_classes,
                 nb_tokens,
                 maxlen,
                 embeddings=None,
                 embedding_dim=64,
                 embed_dropout_rate=0.25,
                 drop_rate=0.5,
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
        self.embed_dropout_rate = embed_dropout_rate
        self.drop_rate = drop_rate
        self.nb_filters = nb_filters
        self.conv_kernel_size = conv_kernel_size
        self.dilation_rate = dilation_rate
        self.repeat_times = repeat_times
        self.use_crf = use_crf
        if embeddings is not None:
            self.token_embeddings = [embeddings]
        else:
            self.token_embeddings = None
        self.invalid_params = {}

    def forward(self):
        word_ids = Input(shape=(self.maxlen,), dtype='int32', name='token')
        input_data = [word_ids]
        embed = Token_Embedding(word_ids, self.nb_tokens, self.embedding_dim,
                                self.token_embeddings, False, self.maxlen,
                                self.embed_dropout_rate, name='token_embeddings')
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
        if self.drop_rate > 0:
            enc = Dropout(self.drop_rate)(dilation_conv)

        outputs, self._loss, self._acc = sl_output_logits(
            enc, self.nb_classes, self.use_crf)
        self.model = Model(inputs=input_data, outputs=outputs)

    def get_loss(self):
        return self._loss

    def get_metrics(self):
        return self._acc
