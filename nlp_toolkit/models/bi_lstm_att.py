from nlp_toolkit.models import Base_Model
from nlp_toolkit.modules.attentions import Attention
from nlp_toolkit.modules.token_embedders import Token_Embedding
from nlp_toolkit.modules.logits import tc_output_logits
from nlp_toolkit.modules.custom_loss import custom_binary_crossentropy, custom_categorical_crossentropy
from keras.layers import Input, Activation
from keras.layers import LSTM, Bidirectional
from keras.layers.merge import concatenate
from keras.models import Model


class bi_lstm_attention(Base_Model):
    """
    Model is modified from DeepMoji.

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
            self.token_embeddings = [embeddings]
        else:
            self.token_embeddings = None
        self.embed_dropout_rate = embed_dropout_rate
        self.final_dropout_rate = final_dropout_rate
        self.return_attention = return_attention
        self.attention_layer = Attention(
            attention_dim=attention_dim,
            return_attention=return_attention, name='attlayer')

        self.invalid_params = {'attention_layer'}

    def forward(self):
        model_input = Input(shape=(self.maxlen,), dtype='int32', name='token')
        x = Token_Embedding(model_input, self.nb_tokens, self.embedding_dim,
                            self.token_embeddings, True, self.maxlen,
                            self.embed_dropout_rate, name='token_embeddings')
        x = Activation('tanh')(x)

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
