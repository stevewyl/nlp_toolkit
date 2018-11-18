"""
Text preprocess utilties
"""

import re
import sys
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
        classification: text\tlabel
        sequence labeling: token_1###label_1\ttoken_2###label_2\t... \ttoken_n###label_n
        language model: token_1 token_2 ... token_n
    """

    def __init__(self, mode, fname='', tran_fname='',
                 config=None, task_type=None, use_radical=False):
        self.fname = fname
        self.use_radical = use_radical
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
            self.data_config = config['data']
            self.basic_token = self.data_config['basic_token']
            if self.basic_token == 'word':
                self.max_tokens = self.data_config['max_words']
                if task_type == 'sequence_labeling' and (self.data_config['use_seg'] or self.data_config['use_radical']):
                    logger.warning('please set use_seg or use_radical as False if your basic token is word')
                self.inner_char = self.data_config['inner_char']
                self.use_seg = False
                self.use_radical = False
            elif self.basic_token == 'char':
                self.max_tokens = self.data_config['max_chars']
                if self.data_config['inner_char']:
                    logger.warning('please set inner_char as False in config file')
                    self.data_config['inner_char'] = False
                self.inner_char = False
                self.use_seg = self.data_config['use_seg'] 
                self.use_radical = self.data_config['use_radical']
            else:
                logger.error('invalid token type, only support word and char')
                sys.exit()
            self.embed_config = config['embed']
            self.config = config
            if task_type == 'sequence_labeling':
                if self.config['train']['metric'] not in ['f1_seq']:
                    logger.error('sequence labeling task only support f1_seq callback')
                    sys.exit()
            elif task_type == 'classification':
                if self.config['train']['metric'] in ['f1_seq']:
                    logger.error('text classification task not support f1_seq callback')
                    sys.exit()
            self.transformer = IndexTransformer(
                self.task_type, self.max_tokens, self.data_config['max_inner_chars'],
                use_inner_char=self.data_config['inner_char'],
                use_seg=self.use_seg, use_radical=self.use_radical,
                basic_token=self.basic_token)
        elif mode == 'predict':
            if len(tran_fname) > 0:
                logger.info('transformer loaded')
                self.transformer = IndexTransformer.load(tran_fname)
                self.basic_token = self.transformer.basic_token
            else:
                logger.error("please input the transformer's filepath")
                sys.exit()
        if self.use_radical:
            self.radical_dict = {line.split()[0]: line.split()[1].strip()
                                 for line in open('./data/radical.txt', encoding='utf8')}
        self.mode = mode
        if fname:
            self.load_data()
        else:
            self.texts = []
            self.labels = []
        self.html_texts = re.compile(r'('+'|'.join(REGEX_STR)+')', re.UNICODE)

    def load_data(self):
        if self.mode == 'train':
            if self.task_type == 'classification':
                self.texts, self.labels = zip(
                    *[line.strip().split('\t') for line in open(self.fname, 'r', encoding='utf8')])
            elif self.task_type == 'sequence_labeling':
                data = (line.strip() for line in open(self.fname, 'r', encoding='utf8'))
                if self.data_config['format'] == 'basic':
                    self.texts, self.labels = zip(
                        *[zip(*[item.rsplit('###') for item in line.split('\t')]) for line in data])
                elif self.data_config['format'] == 'conll':
                    self.texts, self.labels = self.process_conll(data)
                else:
                    logger.warning('invalid data format for sequence labeling task')
                    sys.exit()
        elif self.mode == 'predict':
            self.texts = [line.strip() for line in open(self.fname, 'r', encoding='utf8')]
        logger.info('data loaded')

    def add(self, line: Dict[str, str]):
        if self.mode == 'train':
            if self.task_type == 'classification':
                self.texts.append(line['text'].strip())
                self.labels.append(line['label'])
            elif self.task_type == 'sequence_labeling':
                t = line['text'].strip().split()
                l = line['label'].strip().split()
                assert len(t) == len(l)
                self.texts.append(t)
                self.labels.append(l)
        elif self.mode == 'predict':
            self.texts.append(line['text'].strip())

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

    def clean(self, line):
        line = re.sub(r'\[[\u4e00-\u9fa5a-z]{1,4}\]|\[aloha\]|\t', '', line)
        line = re.sub(EMOJI_UNICODE, '', line)
        line = re.sub(self.html_texts, '', line)
        # line = re.sub('c#{1}', 'c_sharp', line)
        if re.search(r'[\u4300-\u9fa5]+', line):
            line = HanziConv.toSimplified(line)
            return re.sub(' {2,}', ' ', line).lower()
        else:
            return None

    # 转折句简单切分
    def adv_split(self, line):
        return re.sub('(' + '|'.join(ADVERSATIVES) + ')', r'<turn>', line)

    def transform(self):
        if self.task_type == 'classification':
            self.texts = (self.clean(line) for line in self.texts)
            if self.mode == 'train':
                x, y = zip(*[(x1.split(' '), y1.split(' '))
                             for x1, y1 in zip(self.texts, self.labels) if x1])
            elif self.mode == 'predict':
                x = [x1.split(' ') for x1 in self.texts]
            if self.basic_token == 'char':
                x = [word2char(item, task_type='classification') for item in x]
        elif self.task_type == 'sequence_labeling':
            if self.mode == 'train':
                x = self.texts
                y = self.labels
                if self.basic_token == 'char':
                    new_results = [word2char(x1, x2, self.task_type, self.use_seg, self.use_radical)
                                   for x1, x2 in zip(self.texts, self.labels)]
                    x = [item['token'] for item in new_results]
                    y = [item['label'] for item in new_results]
            elif self.mode == 'predict':
                x = self.texts
                if self.basic_token == 'char':
                    new_results = [word2char(x, None, 'sequence_labeling', self.use_seg, self.use_radical)
                                   for x in self.texts]
                    x = [item['token'] for item in new_results]
            if self.use_seg:
                seg_tags = [item['seg'] for item in new_results]
            if self.use_radical:
                radicals = [item['radical'] for item in new_results]

        x = {'token': x}
        if self.task_type == 'sequence_labeling':
            if self.use_seg:
                x['seg'] = seg_tags
            if self.use_radical:
                x['radical'] = radicals

        if self.mode == 'train':
            self.config['mode'] = self.mode
            x, y = self.transformer.fit_transform(x, y)
            logger.info('texts and labels have been transformed to number index')
            embed = {}
            if self.embed_config['pre']:
                token_embed, dim = load_vectors(
                    self.embed_config[self.basic_token]['path'], self.transformer._token_vocab)
                embed[self.basic_token] = token_embed
                logger.info('Loaded Pre_trained Embeddings')
            else:
                logger.info('Use Embeddings from Straching')
                dim = self.embed_config[self.basic_token]['dim']
                embed[self.basic_token] = None
            # update config
            self.config['nb_classes'] = self.transformer.label_size
            self.config['nb_tokens'] = self.transformer.token_vocab_size
            self.config['extra_features'] = []
            if self.inner_char:
                self.config['nb_char_tokens'] = self.transformer.char_vocab_size
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
            self.config['maxlen'] = self.transformer.max_tokens
            self.config['task_type'] = self.task_type
            return x, y, self.config

        elif self.mode == 'predict':
            x_seq = self.transformer.transform(x)
            return x_seq
