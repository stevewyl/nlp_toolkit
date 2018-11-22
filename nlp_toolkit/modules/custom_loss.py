'''
custom loss functions
'''

from keras import backend as K


def custom_binary_crossentropy(y_true, y_pred):
    return K.mean(K.binary_crossentropy(y_true, y_pred[:, :2]), axis=-1)


def custom_categorical_crossentropy(y_true, y_pred, n):
    return K.categorical_crossentropy(y_true, y_pred[:, :n])
