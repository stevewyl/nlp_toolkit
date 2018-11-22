import gc
import logging
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from nlp_toolkit.classifier import Classifier
from nlp_toolkit.labeler import Labeler
from nlp_toolkit.data import Dataset

logging.basicConfig(level=logging.INFO)

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
set_session(tf.Session(config=tf_config))

gc.disable()
