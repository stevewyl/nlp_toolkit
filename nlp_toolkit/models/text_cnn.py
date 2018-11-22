from nlp_toolkit.models import Base_Model
from nlp_toolkit.modules.logits import tc_output_logits
from nlp_toolkit.modules.token_embedders import Token_Embedding
from keras.layers import Input, Dense, Flatten, Dropout
from keras.layers import Conv1D, MaxPooling1D
from keras.layers.merge import concatenate
from keras.models import Model


class textCNN(Base_Model):
    """
    The known Kim CNN model used in text classification.
    It use mulit-channel CNN to encode texts
    """

    def __init__(self, nb_classes, nb_tokens, maxlen,
                 embedding_dim=256, embeddings=None, embed_l2=1E-6,
                 conv_kernel_size=[3, 4, 5], pool_size=[2, 2, 2],
                 nb_filters=128, fc_size=128,
                 embed_dropout_rate=0.25, final_dropout_rate=0.5):
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
        self.embed_dropout_rate = embed_dropout_rate

        # core layer: multi-channel cnn-pool layers
        self.cnn_list = [Conv1D(
            nb_filters, f, padding='same', name='conv_%d' % k) for k, f in enumerate(conv_kernel_size)]
        self.pool_list = [MaxPooling1D(p, name='pool_%d' % k)
                          for k, p in enumerate(pool_size)]
        self.fc = Dense(fc_size, activation='relu',
                        kernel_initializer='he_normal')
        if embeddings is not None:
            self.token_embeddings = [embeddings]
        else:
            self.token_embeddings = None
        self.invalid_params = {'cnn_list', 'pool_list', 'fc'}

    def forward(self):
        model_input = Input(shape=(self.maxlen,), dtype='int32', name='token')
        x = Token_Embedding(model_input, self.nb_tokens, self.embedding_dim,
                            self.token_embeddings, False, self.maxlen,
                            self.embed_dropout_rate, name='token_embeddings')
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
