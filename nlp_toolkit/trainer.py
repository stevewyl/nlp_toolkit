"""
Trainer Class: define the training process
"""

import os
import numpy as np
from keras.optimizers import Adam, Nadam
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from nlp_toolkit.callbacks import get_callbacks
from nlp_toolkit.utilities import logger
from nlp_toolkit.sequence import TextSequence, BucketedSequence


class Trainer(object):
    """
    Trainer class for all model training
    support single training and n-fold training

    # Arguments:
        1. model: Keras Model object
        2. 
        3. 
    """

    def __init__(self, model,
                 model_name,
                 batch_size=64,
                 max_epoch=25,
                 optimizer=Adam,
                 checkpoint_path='./logs/',
                 early_stopping=True,
                 lrplateau=True,
                 tensorboard=False,
                 train_mode='single',
                 fold_cnt=10,
                 test_size=0.2,
                 shuffle=True,):
        self.model = model
        self.max_epoch = max_epoch
        self.batch_size = batch_size
        self.model_name = model_name
        self.optimizer = optimizer
        self.test_size = test_size
        self.train_mode = train_mode
        self.fold_cnt = fold_cnt
        self.shuffle = shuffle

    def train(self, x, y, transformer, seq, verbose=1):
        x_len = x['len']
        x = x['word']
        if self.train_mode == 'single':
            self.model.build()
            logger.info('%s model structure...' % self.model_name)
            self.model.model.summary()

            indices = np.random.permutation(x.shape[0])
            cut_point = x.shape[0] * (1 - self.test_size)
            train_idx, valid_idx = indices[:cut_point], indices[cut_point:]
            x_train, x_valid = x[train_idx, :], x[valid_idx, :]
            y_train, y_valid = y[train_idx, :], y[valid_idx, :]
            x_len_train, x_len_valid = x_len[train_idx, :], x_len[valid_idx, :]
            if seq == 'bucket':
                train_batches = BucketedSequence(
                    100, self.batch_size, x_len_train, x_train, y_train)
                valid_batches = BucketedSequence(
                    100, self.batch_size, x_len_valid, x_valid, y_valid)
            elif seq == 'basic':
                train_batches = TextSequence(x_train, y_train, self.batch_size)
                valid_batches = TextSequence(x_valid, y_valid, self.batch_size)

            callbacks = get_callbacks(
                log_dir='./logs/', valid=valid_batches, preprocessor=transformer)

            if transformer.label_size == 2:
                loss_func = 'binary_crossentropy'
            elif transformer.label_size > 2:
                loss_func = 'categorical_crossentropy'
            self.model.model.compile(
                loss=loss_func, optimizer=self.optimizer, metrics=['acc'])

            self.model.model.fit_generator(generator=train_batches,
                                           epochs=self.max_epoch,
                                           callbacks=callbacks,
                                           verbose=verbose,
                                           shuffle=self.shuffle,
                                           validation_data=valid_batches)
            return self.model.model

        elif self.train_mode == 'fold':
            fold_size = len(x) // self.fold_cnt
            scores = {}
            for fold_id in range(self.fold_cnt):
                print('\n------------------------ fold ' + str(fold_id) + '--------------------------------------')

                self.model.build()

                fold_start = fold_size * fold_id
                fold_end = fold_start + fold_size
                if fold_id == fold_size - 1:
                    fold_end = len(x)
                if fold_id == 0:
                    print('%s model structure...' % self.model_name)
                    self.model.model.summary()

                    x_train = np.concatenate([x[:fold_start], x[fold_end:]])
                    y_train = np.concatenate([y[:fold_start], y[fold_end:]])
                    x_valid = x[fold_start:fold_end]
                    y_valid = y[fold_start:fold_end]
