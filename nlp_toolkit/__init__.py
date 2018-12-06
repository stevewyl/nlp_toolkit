import gc
import os
import logging
import numpy as np
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from nlp_toolkit.classifier import Classifier
from nlp_toolkit.labeler import Labeler
from nlp_toolkit.data import Dataset

logging.basicConfig(level=logging.INFO)

os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free > tmp')
memory_gpu = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
gpu_no = str(np.argmax(memory_gpu))
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_no
os.system('rm tmp')

tf_config = tf.ConfigProto()
tf_config.gpu_options.allow_growth = True
set_session(tf.Session(config=tf_config))
logging.info('Choose the most free GPU: %s' % gpu_no)

gc.disable()
