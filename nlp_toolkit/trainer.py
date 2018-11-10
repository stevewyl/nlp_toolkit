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
from nlp_toolkit.layer import custom_binary_crossentropy, custom_categorical_crossentropy


class Trainer(object):
    """
    Trainer class for all model training
    support single training and n-fold training

    # Arguments:
        1. model: Keras Model object
        2. model_name
        3. metric: the main metric used to track model performance on epoch end
        4. batch_size
        5. max_epoch
        6. optimizer
        7. checkpoint_path: the folder path for saving models
        8. early_stopping
        9. lrplateau
        10. tensorboard: whether to open tensorboard to log training process
        11. train_mode: single turn training or n-fold training
        12. fold_cnt: the number of folds
        13. test_size
        14. shuffle: whether to shuffle data between epochs
        15. seq_type: basic iterator or bucketiterator
    # Returns:
        The trained model or average performance of the model
    """

    def __init__(self, model,
                 model_name,
                 task_type,
                 metric='f1',
                 batch_size=64,
                 max_epoch=25,
                 optimizer=Adam(),
                 checkpoint_path='./models/',
                 early_stopping=True,
                 lrplateau=True,
                 tensorboard=False,
                 train_mode='single',
                 fold_cnt=10,
                 test_size=0.2,
                 shuffle=True,):
        self.model = model
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
                self.task_type, 100, self.batch_size,
                x_len_train, x_train, y_train, self.concat)
            valid_batches = BucketIterator(
                self.task_type, 100, self.batch_size,
                x_len_valid, x_valid, y_valid, self.concat)
        elif seq_type == 'basic':
            train_batches = BasicIterator(
                x_train, y_train, self.batch_size, self.concat)
            valid_batches = BasicIterator(
                x_valid, y_valid, self.batch_size, self.concat)
        else:
            logger.warning('invalid data iterator type, only supports "basic" or "bucket"')
        return train_batches, valid_batches

    def train(self, x, y, transformer,
              seq_type='bucket',
              return_attention=False, use_crf=False):
        assert isinstance(x, dict)
        x_len = np.array(x['length'])
        if 'inner_char' in x.keys():
            self.concat = True
            x_char = x['inner_char']
            x_word = x['word']
            x_word = np.expand_dims(x_word, axis=-1)
            x = np.concatenate((x_word, x_char), axis=-1)
        else:
            self.concat = False
            x = x['word']
        if self.train_mode == 'single':
            # model initialization
            self.model.forward()
            logger.info('%s model structure...' % self.model_name)
            self.model.model.summary()

            # split dataset
            indices = np.random.permutation(x.shape[0])
            cut_point = int(x.shape[0] * (1 - self.test_size))
            train_idx, valid_idx = indices[:cut_point], indices[cut_point:]
            x_train, x_valid = x[train_idx, :], x[valid_idx, :]
            y_train, y_valid = y[train_idx, :], y[valid_idx, :]
            x_len_train, x_len_valid = x_len[train_idx], x_len[valid_idx]
            logger.info('train/valid set: {}/{}'.format(x_train.shape[0], x_valid.shape[0]))

            # transform data to sequence data streamer
            train_batches, valid_batches = self.data_generator(
                seq_type,
                x_train, x_valid, y_train, y_valid,
                x_len_train, x_len_valid)

            # define callbacks
            history = History(self.metric)
            self.callbacks = get_callbacks(
                history=history,
                metric=self.metric,
                log_dir=self.checkpoint_path,
                valid=valid_batches,
                transformer=transformer,
                attention=return_attention)

            # model compile
            self.model.model.compile(
                loss=self.model.get_loss(),
                optimizer=self.optimizer,
                metrics=self.model.get_metrics())

            # save transformer and model parameters
            if not self.checkpoint_path.exists():
                self.checkpoint_path.mkdir()
            transformer.save(self.checkpoint_path / 'transformer.h5')
            self.model.save_params(
                self.checkpoint_path / 'model_parameters.json')
            logger.info('saving model parameters and transformer to {}'.format(
                self.checkpoint_path))

            # actual training start
            self.model.model.fit_generator(
                generator=train_batches,
                epochs=self.max_epoch,
                callbacks=self.callbacks,
                shuffle=self.shuffle,
                validation_data=valid_batches)
            print('best {}: {:04.2f}'.format(self.metric, max(history.metrics)))
            return self.model.model, history

        elif self.train_mode == 'fold':
            fold_size = len(x) // self.fold_cnt
            scores = []
            logger.info('%d-fold starts!' % self.fold_cnt)
            for fold_id in range(self.fold_cnt):
                print('\n------------------------ fold ' + str(fold_id) + '------------------------')

                self.model.forward()

                fold_start = fold_size * fold_id
                fold_end = fold_start + fold_size
                if fold_id == fold_size - 1:
                    fold_end = len(x)
                if fold_id == 0:
                    logger.info('%s model structure...' % self.model_name)
                    self.model.model.summary()

                x_train = np.concatenate([x[:fold_start], x[fold_end:]])
                x_len_train = np.concatenate(
                    [x_len[:fold_start], x_len[fold_end:]])
                y_train = np.concatenate([y[:fold_start], y[fold_end:]])
                x_valid = x[fold_start:fold_end]
                x_len_valid = x_len[fold_start:fold_end]
                y_valid = y[fold_start:fold_end]
                train_batches, valid_batches = self.data_generator(
                    seq_type,
                    x_train, x_valid, y_train, y_valid,
                    x_len_train, x_len_valid)

                history = History(self.metric)
                self.callbacks = get_callbacks(
                    history=history, metric=self.metric,
                    valid=valid_batches, transformer=transformer,
                    attention=return_attention)

                self.model.model.compile(
                    loss=self.model.get_loss(),
                    optimizer=self.optimizer,
                    metrics=self.model.get_metrics())

                self.model.model.fit_generator(
                    generator=train_batches,
                    epochs=self.max_epoch,
                    callbacks=self.callbacks,
                    shuffle=self.shuffle,
                    validation_data=valid_batches)
                scores.append(max(history.metrics))

            logger.info('training finished! The mean {} scores: {:4.2f}(±{:4.2f})'.format(
                self.metric, np.mean(scores), np.std(scores)))
