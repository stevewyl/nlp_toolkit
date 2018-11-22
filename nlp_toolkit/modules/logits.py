"""
common output layers for different tasks
"""

from keras_contrib.layers import CRF
from keras.layers import Dense, Dropout
from keras.regularizers import l2


def tc_output_logits(x, nb_classes, final_dropout_rate=0):
    if final_dropout_rate != 0:
        x = Dropout(final_dropout_rate)(x)
    if nb_classes > 2:
        activation_func = 'softmax'
    else:
        activation_func = 'sigmoid'
    logits = Dense(nb_classes, kernel_regularizer=l2(0.01),
                   activation=activation_func, name='softmax')(x)
    outputs = [logits]
    return outputs


def sl_output_logits(x, nb_classes, use_crf=True):
    if use_crf:
        crf = CRF(nb_classes, sparse_target=False)
        loss = crf.loss_function
        acc = [crf.accuracy]
        outputs = crf(x)
    else:
        loss = 'categorical_crossentropy'
        acc = ['acc']
        outputs = Dense(nb_classes, activation='softmax')(x)
    return outputs, loss, acc
