import gc
import os
import logging
import numpy as np
import tensorflow as tf
from nlp_toolkit.classifier import Classifier
from nlp_toolkit.labeler import Labeler
from nlp_toolkit.data import Dataset
from nlp_toolkit.config import YParams

logging.basicConfig(level=logging.INFO)

try:
    import GPUtil
    from keras.backend.tensorflow_backend import set_session

    num_all_gpu = len(GPUtil.getGPUs())
    avail_gpu = GPUtil.getAvailable(order='memory')
    num_avail_gpu = len(avail_gpu)

    gpu_no = str(avail_gpu[0])
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_no
    logging.info('Choose the most free GPU: %s, currently not support multi-gpus' % gpu_no)

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    set_session(tf.Session(config=tf_config))

except FileNotFoundError:
    logging.info('nvidia-smi is missing, often means no gpu on this machine. '
                 'fall back to cpu!')

gc.disable()
