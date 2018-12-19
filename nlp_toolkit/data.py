"""
Text preprocess utilties
"""

import re
import os
import sys
from pathlib import Path
from hanziconv import HanziConv
from typing import Dict
from nlp_toolkit.sequence import IndexTransformer
from nlp_toolkit.utilities import load_vectors, logger, word2char

EMOJI_UNICODE = r'[\U0001F600-\U0001F64F\U0001F300-\U0001F5FF\u2600-\u26FF\u2700-\u27BF]'
REGEX_STR = [
    r'转发微博|欢迎转发|^回复|…{2,}|图片评论',  # 微博特定停用词
    r'<[^>]+>',  # HTML标记
    r'/{0,2}@\w+-?\w*[:：]?',  # @-用户
    r'#.+#',  # hash-tags
    # URLs
    r'(?:https?://|www\.)(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+',
    r'\b[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-] +\.[a-zA-Z0-9-.] +\b'  # E-MAIL
]
NEGATIVES = ['不', '没有', '无', '莫', '非', '没']
ADVERSATIVES = ['但是', '但', '然而']
SENT_SEP_LIST = r'[。!！？\?]+'


# TODO
class Dataset(object):
    """
    Clean text for post processing. Contains following steps:
        1. remove sepcific tokens(e.g. weibo emotions, emojis, html tags etc.)
        2. must contain Chinese character
        3. simplify Chinese character
        4. segment words supported by pyhanlp (removed)
    Then transform text and label to number index according to the given task

    Data Foramt by line:
        classification: __label__<class_name_1> __label__<class_name_2> ... <text>
        sequence labeling: token_1###label_1\ttoken_2###label_2\t... \ttoken_n###label_n
        language model: token_1 token_2 ... token_n
    """

    def __init__(self, mode, fname='', tran_fname='',
                 config=None, task_type=None, data_format=''):
        self.mode = mode
        self.fname = fname
        self.inner_char = False
        self.use_seg = False
        self.use_radical = False
        self.radical_dict = None
        
        if data_format != '':
            self.data_format = data_format

        if config:
            self.basic_token = config['data']['basic_token']
        self.html_texts = re.compile(r'('+'|'.join(REGEX_STR)+')', re.UNICODE)

        if task_type:
            if mode == 'train' and config is None:
                logger.error('please specify the config file path')
                sys.exit()
            self.task_type = task_type
        else:
            try:
                self.task_type = re.findall(r'config_(\w+)\.yaml', config)[0]
            except:
                logger.error('please check your config filename')
                sys.exit()

        if mode == 'train':
            if 'data' in config:
                self.config = config
                self.data_config = config['data']
                self.embed_config = config['embed']
                self.data_format = self.data_config['format']
                if self.basic_token == 'word':
                    self.max_tokens = self.data_config['max_words']
                    self.inner_char = self.data_config['inner_char']
                elif self.basic_token == 'char':
                    self.max_tokens = self.data_config['max_chars']
                    if self.task_type == 'sequence_labeling':
                        self.use_seg = self.data_config['use_seg']
                        self.use_radical = self.data_config['use_radical']
                        if self.config['train']['metric'] not in ['f1_seq']:
                            self.config['train']['metric'] = 'f1_seq'
                            logger.warning('sequence labeling task currently only support f1_seq callback')
                    elif self.task_type == 'classification':
                        if self.config['train']['metric'] in ['f1_seq']:
                            self.config['train']['metric'] = 'f1'
                            logger.warning('text classification task not support f1_seq callback, changed to f1')
                else:
                    logger.error('invalid token type, only support word and char')
                    sys.exit()
            else:
                logger.error("please pass in the correct config dict")
                sys.exit()

            if self.basic_token == 'char':
                self.use_seg = config['data']['use_seg']
                self.use_radical = config['data']['use_radical']

            if self.use_radical:
                radical_file = Path(os.path.dirname(
                    os.path.realpath(__file__))) / 'data' / 'radical.txt'
                self.radical_dict = {line.split()[0]: line.split()[1].strip()
                                    for line in open(radical_file, encoding='utf8')}

            self.transformer = IndexTransformer(
                task_type=self.task_type,
                max_tokens=self.max_tokens,
                max_inner_chars=self.data_config['max_inner_chars'],
                use_inner_char=self.inner_char,
                use_seg=self.use_seg,
                use_radical=self.use_radical,
                radical_dict=self.radical_dict,
                basic_token=self.basic_token)

        elif mode != 'train':
            if len(tran_fname) > 0:
                logger.info('transformer loaded')
                self.transformer = IndexTransformer.load(tran_fname)
                self.basic_token = self.transformer.basic_token
                self.use_seg = self.transformer.use_seg
                self.use_radical = self.transformer.use_radical
                self.inner_char = self.transformer.use_inner_char
                self.max_tokens = self.transformer.max_tokens
            else:
                logger.error("please pass in the transformer's filepath")
                sys.exit()

        if fname:
            self.load_data()
            self.fit()
        else:
            self.texts = []
            self.labels = []

    def clean(self, line):
        line = re.sub(r'\[[\u4e00-\u9fa5a-z]{1,4}\]|\[aloha\]', '', line)
        line = re.sub(EMOJI_UNICODE, '', line)
        line = re.sub(self.html_texts, '', line)
        if re.search(r'[\u4300-\u9fa5]+', line):
            line = HanziConv.toSimplified(line)
            return re.sub(' {2,}|\t', ' ', line).lower()
        else:
            return None

    def load_data(self):
        if self.task_type == 'classification':
            self.load_tc_data()
        elif self.task_type == 'sequence_labeling':
            if self.mode != 'predict':
                self.load_sl_data()
            else:
                self.texts = [line.strip().split() for line in open(self.fname, 'r', encoding='utf8')]
        logger.info('data loaded')

    def load_tc_data(self, max_tokens_per_doc=256):
        """
        Reads a data file for text classification. The file should contain one document/text per line.
        The line should have the following format:
        __label__<class_name> <text>
        If you have a multi label task, you can have as many labels as you want at the beginning of the line, e.g.,
        __label__<class_name_1> __label__<class_name_2> <text>
        """
        label_prefix = '__label__'
        self.texts = []
        self.labels = []

        with open(self.fname, 'r', encoding='utf8') as fin:
            for line in fin:
                words = self.clean(line.strip()).split()
                if self.mode != 'predict':
                    if words:
                        nb_labels = 0
                        label_line = []
                        for word in words:
                            if word.startswith(label_prefix):
                                nb_labels += 1
                                label = word.replace(label_prefix, "")
                                label_line.append(label)
                            else:
                                break
                        text = words[nb_labels:]
                        if len(text) > max_tokens_per_doc:
                            text = text[:max_tokens_per_doc]
                        self.texts.append(text)
                        self.labels.append(label_line)
                else:
                    self.texts.append(words)

    def load_sl_data(self):
        """
        Reads a data file for text classification. The file should contain one document/text per line.
        The line should have the following formats:
        1. conll:
            word\ttag
            ...
            word\ttag

            word\ttag
            ...
        2. basic:
            word###tag\tword###tag\t...word###tag
        """
        data = (line.strip() for line in open(self.fname, 'r', encoding='utf8'))
        if self.data_format == 'basic':
            self.texts, self.labels = zip(
                *[zip(*[item.rsplit('###', 1) for item in line.split('\t')]) for line in data])
            self.texts = list(map(list, self.texts))
            self.labels = list(map(list, self.labels))
        elif self.data_format == 'conll':
            self.texts, self.labels = self.process_conll(data)
        else:
            logger.warning('invalid data format for sequence labeling task')
            sys.exit()

    def process_conll(self, data):
        sents, labels = [], []
        tokens, tags = [], []
        for line in data:
            if line:
                token, tag = line.split('\t')
                tokens.append(token)
                tags.append(tag)
            else:
                sents.append(tokens)
                labels.append(tags)
                tokens, tags = [], []
        return sents, labels

    def add(self, line: Dict[str, str]):
        t = line['text'].strip().split()
        if self.mode == 'train':
            l = line['label'].strip().split()
            if self.task_type == 'sequence_labeling':
                assert len(t) == len(l)
            self.texts.append(t)
            self.labels.append(l)
        elif self.mode == 'predict':
            self.texts.append(t)

    # 转折句简单切分
    def adv_split(self, line):
        return re.sub('(' + '|'.join(ADVERSATIVES) + ')', r'<turn>', line)

    def fit(self):
        if self.mode != 'predict':
            if self.basic_token == 'char':
                if self.task_type == 'sequence_labeling':
                    self.texts = [
                        word2char(x, y, self.task_type, self.use_seg, self.radical_dict)
                        for x, y in zip(self.texts, self.labels)]
                    self.texts = {k: [dic[k] for dic in self.texts] for k in self.texts[0]}
                    self.labels = self.texts['label']
                    del self.texts['label']
                else:
                    self.texts = {'token': [word2char(x, task_type=self.task_type) for x in self.texts]}
            else:
                self.texts = {'token': self.texts}
            if self.mode == 'train':
                self.config['mode'] = self.mode
                self.transformer.fit(self.texts['token'], self.labels)
                logger.info('transformer fitting complete')
                embed = {}
                if self.embed_config['pre']:
                    token_embed, dim = load_vectors(
                        self.embed_config[self.basic_token]['path'], self.transformer._token_vocab)
                    embed[self.basic_token] = token_embed
                    logger.info('Loaded Pre_trained Embeddings')
                else:
                    logger.info('Use Embeddings from Straching ')
                    dim = self.embed_config[self.basic_token]['dim']
                    embed[self.basic_token] = None
                # update config
                self.config['nb_classes'] = self.transformer.label_size
                self.config['nb_tokens'] = self.transformer.token_vocab_size
                self.config['extra_features'] = []
                if self.inner_char:
                    self.config['nb_char_tokens'] = self.transformer.char_vocab_size
                else:
                    self.config['nb_char_tokens'] = 0
                    self.config['use_inner_char'] = False
                if self.use_seg:
                    self.config['nb_seg_tokens'] = self.transformer.seg_vocab_size
                    self.config['extra_features'].append('seg')
                    self.config['use_seg'] = self.use_seg
                else:
                    self.config['nb_seg_tokens'] = 0
                    self.config['use_seg'] = False
                if self.use_radical:
                    self.config['nb_radical_tokens'] = self.transformer.radical_vocab_size
                    self.config['extra_features'].append('radical')
                    self.config['use_radical'] = self.use_radical
                else:
                    self.config['nb_radical_tokens'] = 0
                    self.config['use_radical'] = False
                self.config['embedding_dim'] = dim
                self.config['token_embeddings'] = embed[self.basic_token]
                self.config['maxlen'] = self.max_tokens
                self.config['task_type'] = self.task_type
        else:
            if self.basic_token == 'char':
                self.texts = [
                    word2char(x, None, self.task_type, self.use_seg, self.radical_dict)
                    for x in self.texts]
                self.texts = {k: [dic[k] for dic in self.texts]
                              for k in self.texts[0]}
            else:
                self.texts = {'token': self.texts}

        lengths = [len(item) if len(item) <= self.max_tokens else self.max_tokens
                   for item in self.texts['token']]
        self.texts['token'] = list(map(list, self.texts['token']))
        self.texts['token'] = [item + [lengths[idx]] for idx, item in enumerate(self.texts['token'])]


# TODO
class Sentence(object):
    """
    """

    def __init__(self, transformer):
        pass
