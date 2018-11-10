"""
Text preprocess utilties
"""

import re
import sys
from pyhanlp import HanLP
from hanziconv import HanziConv
from nlp_toolkit.sequence import IndexTransformer
from nlp_toolkit.utilities import load_vectors, logger

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


def hanlp_segment(line, res_type='str'):
    res = [term.word for term in HanLP.segment(line)]
    if res_type == 'str':
        return ' '.join(res)
    else:
        return res


# TODO
# 1. sentence split
class Dataset(object):
    """
    Clean text for post processing. Contains following steps:
        1. remove sepcific tokens(e.g. weibo emotions, emojis, html tags etc.)
        2. must contain Chinese character
        3. simplify Chinese character
        4. segment words supported by pyhanlp (optional)
    Then transform text and label to number index according to the given task

    Data Foramt by line:
        classification: text\tlabel
        sequence labeling: token_1 token_2 ... token_n\tlabel_1 label_2 ... label_n
        language model: token_1 token_2 ... token_n
    """

    def __init__(self, fname, task_type, mode,
                 tran_fname='', config=None, segment=True):
        self.fname = fname
        self.task_type = task_type
        if mode == 'train':
            self.data_config = config['data']
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
                self.task_type, self.data_config['max_words'],
                use_inner_char=self.data_config['inner_char'])
        elif mode == 'predict':
            if len(tran_fname) > 0:
                logger.info('transformer loaded')
                self.transformer = IndexTransformer.load(tran_fname)
            else:
                logger.warning("please input the transformer's filepath")
                self.transformer = None
        self.mode = mode
        self.load_data()
        self.segment = segment
        self.html_texts = re.compile(r'('+'|'.join(REGEX_STR)+')', re.UNICODE)

    def load_data(self):
        if self.mode == 'train':
            self.texts, self.labels = zip(
                *[line.strip().split('\t') for line in open(self.fname, 'r', encoding='utf8')])
        elif self.mode == 'predict':
            self.texts = [line.strip() for line in open(self.fname, 'r', encoding='utf8')]
        logger.info('data loaded')

    def clean(self, line):
        line = re.sub(r'\[[\u4e00-\u9fa5a-z]{1,4}\]|\[aloha\]|\t', '', line)
        line = re.sub(EMOJI_UNICODE, '', line)
        line = re.sub(self.html_texts, '', line)
        if re.search(r'[\u4300-\u9fa5]+', line):
            line = HanziConv.toSimplified(line)
            if self.segment:
                line = hanlp_segment(line)
            return re.sub(' {2,}', ' ', line).lower()
        else:
            return None

    def adv_split(self, line):
        return re.sub('(' + '|'.join(ADVERSATIVES) + ')', r'<turn>', line)

    def transform(self):
        if self.task_type == 'classification':
            cleaned_texts = (self.clean(line) for line in self.texts)
        elif self.task_type == 'sequence_labeling':
            cleaned_texts = (line for line in self.texts)

        if self.mode == 'train':
            self.config['mode'] = self.mode
            x, y = zip(*[(x1.split(' '), y1.split(' ')) for x1, y1 in zip(cleaned_texts, self.labels) if x1])
            x, y = self.transformer.fit_transform(x, y)
            logger.info('texts and labels transformed to number index')
            embed = {}
            if self.embed_config['pre']:
                word_embed, dim = load_vectors(
                    self.embed_config[self.task_type]['word'], self.transformer._word_vocab)
                embed['word'] = word_embed
                logger.info('Loaded Pre_trained Embeddings')
            else:
                logger.info('Use Embeddings from Straching')
                if self.task_type == 'classification':
                    dim = 256
                elif self.task_type == 'sequence_labeling':
                    dim = 64
                embed['word'] = None
            # update config
            self.config['nb_classes'] = self.transformer.label_size
            self.config['nb_tokens'] = self.transformer.word_vocab_size
            self.config['nb_char_tokens'] = self.transformer.char_vocab_size
            self.config['embedding_dim'] = dim
            self.config['word_embeddings'] = embed['word']
            self.config['maxlen'] = self.transformer.max_words
            self.config['task_type'] = self.task_type
            return x, y, self.config

        elif self.mode == 'predict':
            x = [x1.split(' ') for x1 in cleaned_texts if x1]
            x_seq = self.transformer.transform(x)
            return x_seq


# more detail
def sequence_process(x, y):
    pass
