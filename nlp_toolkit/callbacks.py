"""
Different kinds of callbacks during model training
"""

import numpy as np
from pathlib import Path
from seqeval.metrics import accuracy_score
from seqeval.metrics import f1_score as f1_seq_score
from seqeval.metrics import classification_report as sequence_report
from sklearn.metrics import confusion_matrix, f1_score, classification_report
from keras.callbacks import ModelCheckpoint, Callback, EarlyStopping, ReduceLROnPlateau


class Top_N_Acc(Callback):
    """
    Evaluate model with top n label acc at each epoch
    """

    def __init__(self, seq, top_n=5, attention=False, transformer=None):
        super(Top_N_Acc, self).__init__()
        self.seq = seq
        self.top_n = top_n
        self.t = transformer
        self.attention = attention

    def on_epoch_end(self, epoch, logs={}):
        label_true, label_pred = [], []
        for i in range(len(self.seq)):
            x_true, y_true = self.seq[i]
            y_pred = self.model.predict_on_batch(x_true)
            if self.attention:
                y_pred = y_pred[:, :self.t.label_size]
            y_true = self.t.inverse_transform(y_true)
            y_pred = self.t.inverse_transform(y_pred, 5)
            label_true.extend(y_true)
            label_pred.extend(y_pred)
        assert len(label_pred) == len(label_true)
        correct = 0
        for i in range(len(label_pred)):
            if label_true[i] in label_pred[i]:
                correct += 1
        top_n_acc = correct / len(label_pred)
        print(' - top_{}_acc: {:04.2f}'.format(self.top_n, top_n_acc * 100))
        logs['acc_%d' % self.top_n] = np.float64(top_n_acc)


class F1score(Callback):
    """
    Evaluate classification model with f1 score at each epoch
    """

    def __init__(self, seq, attention=False, transformer=None):
        super(F1score, self).__init__()
        self.seq = seq
        self.t = transformer
        self.attention = attention

    def on_epoch_end(self, epoch, logs={}):
        label_true, label_pred = [], []
        for i in range(len(self.seq)):
            x_true, y_true = self.seq[i]

            y_true = np.argmax(y_true, -1)
            y_pred = self.model.predict_on_batch(x_true)
            if self.attention:
                y_pred = y_pred[:, :self.t.label_size]
            y_pred = np.argmax(y_pred, -1)
            label_true.extend(y_true)
            label_pred.extend(y_pred)

        assert len(label_pred) == len(label_true)
        f1 = self._calc_f1(label_true, label_pred)
        assert f1.shape[0] == self.t.label_size
        for i in range(f1.shape[0]):
            label = self.t._label_vocab._id2token[i]
            print(label, '- f1: {:04.2f}'.format(f1[i] * 100))
        # print(classification_report(label_true, label_pred))
        logs['f1'] = f1_score(label_true, label_pred, average='weighted')

    def _calc_f1(self, y_true, y_pred):
        cm = confusion_matrix(y_true, y_pred)
        correct_preds = np.diagonal(cm)
        r = correct_preds / np.sum(cm, axis=1)
        p = correct_preds / np.sum(cm, axis=0)
        f1 = 2 * p * r / (p + r)
        return f1


class F1score_seq(Callback):
    """
    Evaluate sequence labeling model with f1 score at each epoch
    """

    def __init__(self, seq, transformer=None):
        super(F1score_seq, self).__init__()
        self.seq = seq
        self.t = transformer

    def get_lengths(self, y_true):
        lengths = []
        for y in np.argmax(y_true, -1):
            try:
                i = list(y).index(0)
            except ValueError:
                i = len(y)
            lengths.append(i)
        return lengths

    def on_epoch_end(self, epoch, logs={}):
        label_true, label_pred = [], []
        for i in range(len(self.seq)):
            x_true, y_true = self.seq[i]
            lengths = self.get_lengths(y_true)
            y_pred = self.model.predict_on_batch(x_true)
            y_true = self.t.inverse_transform(y_true, lengths)
            y_pred = self.t.inverse_transform(y_pred, lengths)
            label_true.extend(y_true)
            label_pred.extend(y_pred)
        acc = accuracy_score(label_true, label_pred)
        f1 = f1_seq_score(label_true, label_pred)
        print(' - acc: {:04.2f}'.format(acc * 100))
        print(' - f1: {:04.2f}'.format(f1 * 100))
        print(sequence_report(label_true, label_pred))
        logs['f1_seq'] = np.float64(f1)
        logs['seq_acc'] = np.float64(acc)


class History(Callback):
    def __init__(self, metric):
        self.metric = metric

    def on_train_begin(self, logs={}):
        self.losses = []
        self.acc = []
        self.metrics = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))
        self.acc.append(logs.get('acc'))

    def on_epoch_end(self, epoch, logs={}):
        self.metrics.append(logs.get(self.metric))


def get_callbacks(history=None, log_dir=None, valid=None, metric='f1',
                  transformer=None, early_stopping=True, patiences=3,
                  LRPlateau=True, top_n=5, attention=False):
    """
    Define list of callbacks for Keras model
    """
    callbacks = []
    if valid is not None:
        if metric == 'top_n_acc':
            print('mointor training process using top_%d_acc score' % top_n)
            callbacks.append(Top_N_Acc(valid, top_n, attention, transformer))
        elif metric == 'f1':
            print('mointor training process using f1 score')
            callbacks.append(F1score(valid, attention, transformer))
        elif metric == 'f1_seq':
            print('mointor training process using f1 score and label acc')
            callbacks.append(F1score_seq(valid, transformer))

    if log_dir:
        path = Path(log_dir)
        if not path.exists():
            print('Successfully made a directory: {}'.format(log_dir))
            path.mkdir()

        file_name = '_'.join(
            ['model_weights', '{epoch:02d}', '{val_acc:2.4f}', '{%s:2.4f}' % metric]) + '.h5'
        weight_file = path / file_name
        save_model = ModelCheckpoint(str(weight_file),
                                     monitor=metric,
                                     verbose=1,
                                     save_best_only=True,
                                     save_weights_only=True,
                                     mode='max')
        callbacks.append(save_model)

    if early_stopping:
        print('using Early Stopping')
        callbacks.append(EarlyStopping(
            monitor=metric, patience=patiences, mode='max'))

    if LRPlateau:
        print('using Reduce LR On Plateau')
        callbacks.append(ReduceLROnPlateau(
            monitor=metric, factor=0.2, patience=patiences-2, min_lr=0.00001))

    if history:
        print('tracking loss history and metrics')
        callbacks.append(history)

    return callbacks
