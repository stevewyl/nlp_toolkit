from keras.engine import Layer
from keras import backend as K
from keras.layers import Embedding, Dropout, SpatialDropout1D, TimeDistributed
from keras.regularizers import L1L2


def Token_Embedding(x, input_dim, output_dim, embed_weights=None,
                    mask_zero=False, input_length=None, dropout_rate=0,
                    embed_l2=1E-6, name='', time_distributed=False, **kwargs):
    """
    Basic token embedding layer, also included some dropout layer.
    """
    embed_reg = L1L2(l2=embed_l2) if embed_l2 != 0 else None
    embed_layer = Embedding(input_dim=input_dim,
                            output_dim=output_dim,
                            weights=embed_weights,
                            mask_zero=mask_zero,
                            input_length=input_length,
                            embeddings_regularizer=embed_reg,
                            name=name)
    if time_distributed:
        embed = TimeDistributed(embed_layer)(x)
    else:
        embed = embed_layer(x)
    # entire embedding channels are dropped out instead of the
    # normal Keras embedding dropout, which drops all channels for entire words
    # many of the datasets contain so few words that losing one or more words can alter the emotions completely
    if dropout_rate != 0:
        embed = SpatialDropout1D(dropout_rate)(embed)
    return embed
