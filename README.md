# nlp_toolkit

中文NLP基础工具箱，包括以下任务：例如文本分类、序列标注等。

本仓库复现了一些近几年比较火的nlp论文。所有的代码是基于keras开发的。

不到10行代码，你就可以快速训练一个文本分类模型或序列标注模型。

## 安装

```bash
git clone https://github.com/stevewyl/nlp_toolkit
cd nlp_toolkit

# 只使用CPU
pip install -r requirements.txt

# 使用GPU
pip install -r requirements-gpu.txt

# 如果keras_contrib安装失败
pip install git+https://www.github.com/keras-team/keras-contrib.git
```

## 使用方法

本仓库的框架图：

![framework](./images/framework.jpg)

主要由以下几大模块组成：

1. Dataset：处理文本和标签数据为适合模型输入的格式，主要进行的处理操作有清理、分词、index化

2. Model Zoo & Layer：近几年在该任务中常用的模型汇总及一些Keras的自定义层
   
   自定义层有如下：

   * 通用注意力层
  
   * 多头注意力层

   * 位置嵌入层

3. Trainer：定义模型的训练流程，支持bucket序列、自定义callbacks和N折交叉验证

    * bucket序列：通过将相似长度的文本放入同一batch来减小padding的多余计算来实现模型训练的加速，在文本分类任务中，能够对RNN网络提速2倍以上（暂时不支持含有Flatten层的网络）
  
    * callbacks：通过自定义回调器来控制训练流程，目前预设的回调器有提前终止训练，学习率自动变化，更丰富的评估函数等

    * N折交叉验证：支持交叉验证来考验模型的真实能力

4. Classifier & Sequence Labeler：封装类，支持不同的训练任务

简单的用法如下：

```python
from nlp_toolkit import Dataset, Classifier, Labeler
import yaml

config = yaml.load(open('your_config.yaml'))

# 分类任务
dataset = Dataset(fname='your_data.txt', task_type='classification', mode='train', config=config)
x, y, new_config = dataset.transform()
text_classifier = Classifier(config=new_config, model_name='multi_head_self_att', seq_type='bucket', transformer=dataset.transformer)
trained_model = text_classifier.train(x, y)

# 序列标注任务
dataset = Dataset(fname='your_data.txt', task_type='seq_label', mode='train', config=config)
x, y, new_config = dataset.transform()
seq_labeler = Labeler(config=new_config, model_name='word_rnn', seq_type='bucket',transformer=dataset.transformer)
trained_model = seq_labeler.train(x, y)

# 预测
```

更多使用细节，请阅读**examples**文件夹中的Jupyter Notebook

### 数据格式

1. 文本分类：每一行为文本+标签，\t分割（暂时不支持多标签任务）

    例如 “公司目前地理位置不太理想， 离城市中心较远点。\tneg”

2. 序列标注：每一行为文本+标签，\t分割，文本序列和标签序列一一对应，以空格分割

    例如 “目前 公司 地理 位置 不 太 理想\tO O B-Chunk I-Chunk O O O”

    标签含义（这里以chunk为例）：

    * O：普通词
    * B-Chunk：表示chunk词的开始
    * I-Chunk：表示chunk词的中间
    * E-Chunk：表示chunk词的结束

    建议：文本序列以短句为主，针对标注实体的任务，最好保证每行数据中有实体词（即非全O的序列）

3. 预测：不同任务每一行均为文本

### 配置文件

train: 表示训练过程中的参数，包括batch大小，epoch数量，训练模式等

data: 表示数据预处理的参数，包括最大词数和字符数，是否使用词内部字符序列，是否开启分词

embed: 词向量，pre表示是否使用预训练词向量

剩下的模块对应不同的模型超参数

具体细节可查看配置文件注释

## 模型

### 文本分类

1. 双层双向LSTM + Attention 🆗

    [DeepMoji](https://arxiv.org/abs/1708.00524)一文中所采用的的模型框架，本仓库中对attention层作了扩展

    对应配置文件名称：bi_lstm_att

2. [Transformer](http://papers.nips.cc/paper/7181-attention-is-all-you-need) 🆗

    采用Transformer中的多头自注意力层来表征文本信息，详细的细节可阅读此[文章](https://kexue.fm/archives/4765)

    对应配置文件名称：multi_head_self_att

3. [TextCNN](https://arxiv.org/abs/1408.5882) 🆗

    CNN网络之于文本分类任务的开山之作，在过去几年中经常被用作baseline，详细的细节可阅读此[文章](http://www.wildml.com/2015/12/implementing-a-cnn-for-text-classification-in-tensorflow/)

    对应配置文件名称：text_cnn

4. [DPCNN](http://www.aclweb.org/anthology/P17-1052)

    通过不断加深CNN网络来获取更好的文本表征

5. [HAN](https://www.cs.cmu.edu/~hovy/papers/16HLT-hierarchical-attention-networks.pdf)

    使用attention机制的文档分类模型

### 序列标注

1. [WordRNN](https://arxiv.org/abs/1707.06799) 🆗

    Baseline模型，文本序列经过双向LSTM后，由CRF层编码作为输出

    对应配置文件名称：word_rnn

2. [CharRNN](https://pdfs.semanticscholar.org/b944/5206f592423f0b2faf05f99de124ccc6aaa8.pdf)

    基于汉语的特点，在字符级别的LSTM信息外，拼接偏旁部首，分词，Ngram信息

3. [InnerChar](https://arxiv.org/abs/1611.04361) 🆗

    基于另外一篇[论文](https://arxiv.org/abs/1511.08308)，扩展了本文的模型，使用bi-lstm或CNN在词内部的char级别进行信息的抽取，然后与原来的词向量进行concat或attention计算

    对应配置文件名称：word_rnn，并设置配置文件data模块中的inner_char为True

4. [IDCNN](https://arxiv.org/abs/1702.02098) 🆗

    膨胀卷积网络，在保持参数量不变的情况下，增大了卷积核的感受野，详细的细节可阅读此[文章](http://www.crownpku.com//2017/08/26/%E7%94%A8IDCNN%E5%92%8CCRF%E5%81%9A%E7%AB%AF%E5%88%B0%E7%AB%AF%E7%9A%84%E4%B8%AD%E6%96%87%E5%AE%9E%E4%BD%93%E8%AF%86%E5%88%AB.html)

    对应配置文件名称：idcnn

## 性能

### 文本分类

Model                   | 10-fold_f1   | Model Size   | Time per epoch
----------------------- | :------:     | :----------: | :-------------:
Bi-LSTM Attention       |              |              | 
Transformer             |              |              |
TextCNN                 |              |              |
DPCNN                   |              |              |
HAN                     |              |              |

### 序列标注

Model                   | 10-fold_f1   | Model Size   | Time per epoch
----------------------- | :------:     | :----------: | :-------------:
Baseline(WordRNN)       |              |              | 
WordRNN + InnerChar     |              |              |
CharRNN                 |              |              |
IDCNN                   |              |              |

## To-Do列表

1. 句子切分模块

2. 加入更多SOTA的模型（例如，BERT）

3. 增加语言模型的训练

4. 支持自定义模块

5. 为每个模型生成一份专属的配置文件

## 感谢

* 数据流模块部分代码借鉴于此： https://github.com/Hironsan/anago/

* 序列标注任务的评估函数来源于此： https://github.com/chakki-works/seqeval
  
* bucket序列化代码来自：https://github.com/tbennun/keras-bucketed-sequence

* 多头注意力层和位置嵌入层代码来自：https://github.com/bojone/attention

## 联系方式

联系人：王奕磊

📧 邮箱：stevewyl@163.com

微信：Steve_1125

--------------------

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

The frameword of this repository：

![framework](./images/framework.jpg)

Following modules are included in：

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
dataset = Dataset(fname='your_data.txt', task_type='seq_label', mode='train', config=config)
x, y, config = dataset.transform()
seq_labeler = Labeler(config=config, model_name='word_rnn', seq_type='bucket',,transformer=dataset.transformer)
trained_model = seq_labeler.train(x, y)

# predict
trained_model
```

For more details, please read the jupyter notebooks in **examples** folder

### Data Format


## Models

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

E-mail: stevewyl@163.com

WeChat: Steve_1125