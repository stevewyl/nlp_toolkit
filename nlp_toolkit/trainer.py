"""
Trainer Class: define the training process
"""

import os
import time
import numpy as np
from pathlib import Path
from keras.optimizers import Adam, Nadam
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_fscore_support
from nlp_toolkit.callbacks import get_callbacks, History
from nlp_toolkit.utilities import logger
from nlp_toolkit.sequence import BasicIterator, BucketIterator
from nlp_toolkit.modules.custom_loss import custom_binary_crossentropy, custom_categorical_crossentropy
from typing import Dict
from copy import deepcopy

np.random.seed(1050)


class Trainer(object):
    """
    Trainer class for all model training
    support single training and n-fold training

    # Arguments:
        1. model: Keras Model object
        2. model_name
        3. task_type: text classification or sequence labeling
        4. metric: the main metric used to track model performance on epoch end
        5. extra_features: besides token features, some useful features will be included
        6. batch_size: minimum batch size
        7. max_epoch: maximum epoch numbers
        8. optimizer: default is Adam
        9. checkpoint_path: the folder path for saving models
        9. early_stopping: whether to use early stopping strategy
        10. lrplateau: whether to use lr lateau strategy
        11. tensorboard: whether to open tensorboard to log training process
        12. nb_bucket: the bucket size
        13. train_mode: single turn training or n-fold training
        14. fold_cnt: the number of folds
        15. test_size: default is 0.2
        16. shuffle: whether to shuffle data between epochs, default is true
        17. patiences: the maximum epochs to stop training when the metric has not been improved

    # Returns:
        The trained model or average performance of the model
    """

    def __init__(self, model,
                 model_name,
                 task_type,
                 metric,
                 batch_size=64,
                 max_epoch=25,
                 optimizer=Adam(),
                 checkpoint_path='./models/',
                 early_stopping=True,
                 lrplateau=True,
                 tensorboard=False,
                 nb_bucket=100,
                 train_mode='single',
                 fold_cnt=10,
                 test_size=0.2,
                 shuffle=True,
                 patiences=3):
        self.single_model = deepcopy(model)
        self.fold_model = deepcopy(model)
        self.model_name = model_name
        self.task_type = task_type
        self.metric = metric
        self.batch_size = batch_size
        self.max_epoch = max_epoch
        self.optimizer = optimizer
        self.test_size = test_size
        self.train_mode = train_mode
        self.fold_cnt = fold_cnt
        self.shuffle = shuffle
        self.nb_bucket = nb_bucket
        self.patiences = patiences
        base_dir = Path(checkpoint_path)
        if not base_dir.exists():
            base_dir.mkdir()
        current_time = time.strftime(
            '%Y%m%d%H%M', time.localtime(time.time()))
        save_dir = self.model_name + '_' + current_time
        self.checkpoint_path = Path(checkpoint_path) / save_dir

    def data_generator(self, seq_type, x_train, x_valid, y_train, y_valid,
                       x_len_train=None, x_len_valid=None,):
        if seq_type == 'bucket':
            logger.info('use bucket sequence to speed up model training')
            train_batches = BucketIterator(
                self.task_type, self.transformer, x_len_train,
                x_train, y_train, self.nb_bucket, self.batch_size)
            valid_batches = BucketIterator(
                self.task_type, self.transformer, x_len_valid,
                x_valid, y_valid, self.nb_bucket, self.batch_size)
        elif seq_type == 'basic':
            train_batches = BasicIterator(
                self.task_type, self.transformer,
                x_train, y_train, self.batch_size)
            valid_batches = BasicIterator(
                self.task_type, self.transformer,
                x_valid, y_valid, self.batch_size)
        else:
            logger.warning('invalid data iterator type, only supports "basic" or "bucket"')
        return train_batches, valid_batches

    def train(self, x_ori, y, transformer,
              seq_type='bucket',
              return_attention=False):
        self.transformer = transformer
        self.feature_keys = list(x_ori.keys())

        if self.train_mode == 'single':
            x = deepcopy(x_ori)
            x_len = [item[-1] for item in x['token']]
            x['token'] = [item[:-1] for item in x['token']]

            # model initialization
            self.single_model.forward()
            logger.info('%s model structure...' % self.model_name)
            self.single_model.model.summary()

            # split dataset
            indices = np.random.permutation(len(x['token']))
            cut_point = int(len(x['token']) * (1 - self.test_size))
            train_idx, valid_idx = indices[:cut_point], indices[cut_point:]
            x_train = {k: [x[k][i] for i in train_idx] for k in self.feature_keys}
            x_valid = {k: [x[k][i] for i in valid_idx] for k in self.feature_keys}
            y_train, y_valid = [y[i] for i in train_idx], [y[i] for i in valid_idx]
            x_len_train, x_len_valid = [x_len[i] for i in train_idx], [x_len[i] for i in valid_idx]
            logger.info(
                'train/valid set: {}/{}'.format(train_idx.shape[0], valid_idx.shape[0]))

            # transform data to sequence data streamer
            train_batches, valid_batches = self.data_generator(
                seq_type,
                x_train, x_valid, y_train, y_valid,
                x_len_train, x_len_valid)

            # define callbacks
            history = History(self.metric)
            self.callbacks = get_callbacks(
                history=history,
                metric=self.metric[0],
                log_dir=self.checkpoint_path,
                valid=valid_batches,
                transformer=transformer,
                attention=return_attention)

            # model compile
            self.single_model.model.compile(
                loss=self.single_model.get_loss(),
                optimizer=self.optimizer,
                metrics=self.single_model.get_metrics())

            # save transformer and model parameters
            if not self.checkpoint_path.exists():
                self.checkpoint_path.mkdir()
            transformer.save(self.checkpoint_path / 'transformer.h5')
            invalid_params = self.single_model.invalid_params
            param_file = self.checkpoint_path / 'model_parameters.json'
            self.single_model.save_params(param_file, invalid_params)
            logger.info('saving model parameters and transformer to {}'.format(
                self.checkpoint_path))

            # actual training start
            self.single_model.model.fit_generator(
                generator=train_batches,
                epochs=self.max_epoch,
                callbacks=self.callbacks,
                shuffle=self.shuffle,
                validation_data=valid_batches)
            print('best {}: {:04.2f}'.format(self.metric[0],
                                             max(history.metrics[self.metric[0]]) * 100))
            return self.single_model.model, history

        elif self.train_mode == 'fold':
            x = deepcopy(x_ori)
            x_len = [item[-1] for item in x['token']]
            x['token'] = [item[:-1] for item in x['token']]
            x_token_first = x['token'][0]

            fold_size = len(x['token']) // self.fold_cnt
            scores = []
            logger.info('%d-fold starts!' % self.fold_cnt)

            for fold_id in range(self.fold_cnt):
                print('\n------------------------ fold ' + str(fold_id) + '------------------------')

                assert x_token_first == x['token'][0]
                model_init = self.fold_model
                model_init.forward()

                fold_start = fold_size * fold_id
                fold_end = fold_start + fold_size
                if fold_id == fold_size - 1:
                    fold_end = len(x)
                if fold_id == 0:
                    logger.info('%s model structure...' % self.model_name)
                    model_init.model.summary()

                x_train = {k: x[k][:fold_start] + x[k][fold_end:] for k in self.feature_keys}
                x_len_train = x_len[:fold_start] + x_len[fold_end:]
                y_train = y[:fold_start] + y[fold_end:]
                x_valid = {k: x[k][fold_start:fold_end] for k in self.feature_keys}
                x_len_valid = x_len[fold_start:fold_end]
                y_valid = y[fold_start:fold_end]

                train_batches, valid_batches = self.data_generator(
                    seq_type,
                    x_train, x_valid, y_train, y_valid,
                    x_len_train, x_len_valid)

                history = History(self.metric)
                self.callbacks = get_callbacks(
                    history=history, metric=self.metric[0],
                    valid=valid_batches, transformer=transformer,
                    attention=return_attention)

                model_init.model.compile(
                    loss=model_init.get_loss(),
                    optimizer=self.optimizer,
                    metrics=model_init.get_metrics())

                model_init.model.fit_generator(
                    generator=train_batches,
                    epochs=self.max_epoch,
                    callbacks=self.callbacks,
                    shuffle=self.shuffle,
                    validation_data=valid_batches)
                scores.append(max(history.metrics[self.metric[0]]))

            logger.info('training finished! The mean {} scores: {:4.2f}(±{:4.2f})'.format(
                self.metric[0], np.mean(scores) * 100, np.std(scores) * 100))
