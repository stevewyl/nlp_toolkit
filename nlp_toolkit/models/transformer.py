from nlp_toolkit.models import Base_Model
from nlp_toolkit.modules.attentions import Self_Attention
from nlp_toolkit.modules.token_embedders import Position_Embedding
from nlp_toolkit.modules.token_embedders import Token_Embedding
from nlp_toolkit.modules.logits import tc_output_logits
from keras.layers import Input, GlobalAveragePooling1D
from keras.models import Model


class Transformer(Base_Model):
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
                 nb_head=8, head_size=16, nb_transformer=2,
                 embedding_dim=256, embeddings=None, embed_l2=1E-6,
                 pos_embed=False, final_dropout_rate=0.15,
                 embed_dropout_rate=0.15):
        self.nb_classes = nb_classes
        self.nb_tokens = nb_tokens
        self.maxlen = maxlen
        self.nb_head = nb_head
        self.head_size = head_size
        self.embedding_dim = embedding_dim
        self.nb_transformer = nb_transformer
        if embeddings is not None:
            self.token_embeddings = [embeddings]
        else:
            self.token_embeddings = None
        self.pos_embed = pos_embed
        self.final_dropout_rate = final_dropout_rate
        self.embed_dropout_rate = embed_dropout_rate
        self.pos_embed_layer = Position_Embedding(name='position_embedding')
        self.transformers = [Self_Attention(
            nb_head, head_size, name='self_attention_%d' % i) for i in range(nb_transformer)]
        self.pool = GlobalAveragePooling1D()
        self.invalid_params = {'pos_embed_layer', 'transformers', 'pool'}

    def forward(self):
        model_input = Input(shape=(self.maxlen,), dtype='int32', name='token')
        x = Token_Embedding(model_input, self.nb_tokens, self.embedding_dim,
                            self.token_embeddings, False, self.maxlen,
                            self.embed_dropout_rate, name='token_embeddings')
        if self.pos_embed:
            x = self.pos_embed_layer(x)
        for i in range(self.nb_transformer):
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
