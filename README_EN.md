# nlp_toolkit

Basic Chinese NLP Toolkits include following tasks, such as text classification, sequence labeling etc.

This repo reproduce some hot nlp papers in recent years. All the code is based on Keras.

Less than 10 lines of code, you can quickly train a text classfication model or sequence labeling model.

## Install

```bash
git clone https://github.com/stevewyl/nlp_toolkit
cd nlp_toolkit

# Use cpu-only
pip install -r requirements.txt

# Use GPU
pip install -r requirements-gpu.txt

# if keras_contrib install fail
pip install git+https://www.github.com/keras-team/keras-contrib.git
```

## Usage

The frameword of this repository:

![framework](./images/framework.jpg)

Following modules are included in:

1. Dataset：Text and label data are processed in a format suitable for model input. The main processing operations are cleaning, word segmentation and indexation.

2. Model Zoo & Layer：The collection of models commonly used in this task in recent years and some custom layers of Keras.

    Customized layers are as followed:

    * Attention

    * Multi-Head Attention

    * Position Embedding

3. Trainer：Define the training process of differnent models, which supports bucket sequence, customed callbacks and N-fold validation training.

    * Bucket Iterator: Accelerate model training by putting texts with similar lengths into the same batch to reduce the extra calculation of padding. In text classification task, it can help speed up RNN by over 2 times. (currently not support for networks with Flatten layer)

    * callbacks: The training process is controlled by custom callbacks. Currently, the preset callbacks include early stopping strategy, automatical learning rate decay, richer evaluation functions and etc.

    * N-fold cross validation: Support cross-validation to test the true capabilities of the model.

4. Classifier & Sequence Labeler：Encapsulates classes that support different training tasks.

Quick start：

```python
from nlp_toolkit import Dataset, Classifier, Labeler
import yaml

config = yaml.load(open('your_config.yaml'))

# text classification task
dataset = Dataset(fname='your_data.txt', task_type='classification', mode='train', config=config)
x, y, config = dataset.transform()
text_classifier = Classifier(config=config, model_name='multi_head_self_att', seq_type='bucket', transformer=dataset.transformer)
trained_model = text_classifier.train(x, y)

# sequence labeling task
dataset = Dataset(fname='your_data.txt', task_type='sequence_labeling', mode='train', config=config)
x, y, config = dataset.transform()
seq_labeler = Labeler(config=config, model_name='word_rnn', seq_type='bucket',,transformer=dataset.transformer)
trained_model = seq_labeler.train(x, y)

# predict (for text classification task)
dataset = Dataset('your_data.txt', task_type='classification', mode='predict', tran_fname='your_transformer.h5', segment=False)
x_seq = dataset.transform()
text_classifier = Classifier('bi_lstm_att', dataset.transformer)
text_classifier.load(weight_fname='your_model_weights.h5', para_fname='your_model_parameters.json')
y_pred = text_classifier.predict(x_seq['word'])
```

For more details, please read the jupyter notebooks in **examples** folder

### Data Format

1. Text Classification: A pretokenised file where each line is in the following format(temporarily does not support multi-label tasks):

    WORD [SPACE] WORD [SPACE] ... [TAB] LABEL \n

    such as "公司 目前 地理 位置 不 太 理想 ， 离 城市 中心 较 远点 。\tneg\n"

2. Sequence Labeling: A pretokenised file where each line is in the following format:

    WORD###TAG [TAB] WORD###TAG [TAB] ..... \n

    such as "目前###O\t公司###O\t地理###B-Chunk\t位置###E-Chunk\t不###O\t太###O\t理想\n"

    label format (chunking as an example):

    * O：common words
    * B-Chunk：indicates the beginning of the chunk word
    * I-Chunk：indicates the middle of the chunk word
    * E-Chunk：indicates the end of the chunk word

    Suggestions: The text sequence is mainly short sentences. For the task of labeling entities, it is best to ensure that there are entity words in each row of data (ie, sequences of non-all Os).

3. Prediction: Each line of different tasks is text.


### Configuration file

Train: indicates the parameters in the training process, including batch size, epoch numbers, training mode, etc.

Data: indicates the parameters of data preprocessing, including the maximum number of words and characters, whether to use the word internal character sequence, whether to use word segmentation

Embed: word vectors, pre indicates whether to use pre-trained word vectors

The remaining modules correspond to different model hyperparameters

See the configuration file comments for details.

## Models

1. Double Bi-LSTM + Attention 🆗

    The model framework used in paper [DeepMoji](https://arxiv.org/abs/1708.00524). The attention layer has been extended in nlp_toolkit.

    Corresponding to the name in the configuration file: bi_lstm_att

2. [Transformer](http://papers.nips.cc/paper/7181-attention-is-all-you-need) 🆗

    Use the multi-head-self-attention layer in Transformer to characterize text information. Read the [article](https://kexue.fm/archives/4765) for details.

    Corresponding to the name in the configuration file: multi_head_self_att

3. [TextCNN](https://arxiv.org/abs/1408.5882) 🆗

    CNN Network's pioneering work on text classification tasks has often been used as a baseline in the past few years. Detailed details can be read in this [Article](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/)

    Corresponding to the name in the configuration file: text_cnn

4. [DPCNN](http://www.aclweb.org/anthology/P17-1052)

    Get better text characterization by continuously deepening the CNN network.

5. [HAN](https://www.cs.cmu.edu/~hovy/papers/16HLT-hierarchical-attention-networks.pdf)

    Document classification model using the attention mechanism

### Sequence Labeling

1. [WordRNN](https://arxiv.org/abs/1707.06799) 🆗

    Baseline model, the text sequence is encoded by the CRF layer after passing through the bidirectional LSTM

    Corresponding to the name in the configuration file: word_rnn

2. [CharRNN](https://pdfs.semanticscholar.org/b944/5206f592423f0b2faf05f99de124ccc6aaa8.pdf)

    Based on the characteristics of Chinese, in addition to the LSTM information at the character level, the radicals, word segmentation, and Ngram information are added.

3. [InnerChar](https://arxiv.org/abs/1611.04361) 🆗

    Based on another [paper](https://arxiv.org/abs/1511.08308), the above model is extended, using bi-lstm or CNN to extract information from the char level inside the word, and then concat with the original word vectors or conduct attention calculation.

    Corresponding to the name in the configuration file: word_rnn, and set the inner_char in the data module in the configuration file to True.

4. [IDCNN](https://arxiv.org/abs/1702.02098) 🆗

    The iterated dilated CNN increases the receptive field of the convolution kernel while keeping the parameter amount constant. The detailed details can be read in this [article](http://www.crownpku.com//2017/08/26/%E7%94%A8IDCNN%E5%92%8CCRF%E5%81%9A%E7%AB%AF%E5%88%B0%E7%AB%AF%E7%9A%84%E4%B8%AD%E6%96%87%E5%AE%9E%E4%BD%93%E8%AF%86%E5%88%AB.html)

    Corresponding to the name in the configuration file: idcnn


### Text Classification

### Sequence Labeling

## Performance

Here list the performace based on following two datasets:

1. Company Pros and Cons: Crawled from Kanzhun.com and Dajie.com, it contains 95K reviews on the pros and cons of different companies.
2. 

### Text Classification

Model                   | 10-fold_f1   | Model Size   | Time per epoch
----------------------- | :------:     | :----------: | :-------------:
Bi-LSTM Attention       |              |              | 
Transformer             |              |              |
TextCNN                 |              |              |
DPCNN                   |              |              |
HAN                     |              |              |

### Sequence Labeling

Model                   | 10-fold_f1   | Model Size   | Time per epoch
----------------------- | :------:     | :----------: | :-------------:
Baseline(WordRNN)       |              |              | 
WordRNN + InnerChar     |              |              |
CharRNN                 |              |              |
IDCNN                   |              |              |

## To-Do List

1. Sentence split module

2. Add more SOTA model(such as BERT)

3. Support for training language model

4. Support for customized moudle

5. Generate a unique configuration file for each model

## Acknowledgments

* The preprocessor part is derived from https://github.com/Hironsan/anago/
* The evaluations for sequence labeling are based on a modified version of https://github.com/chakki-works/seqeval
* Bucket sequence are based on https://github.com/tbennun/keras-bucketed-sequence
* Multi-head attention and position embedding are from: https://github.com/bojone/attention

## Contact
Contact: Yilei Wang

📧 E-mail: stevewyl@163.com

WeChat: Steve_1125