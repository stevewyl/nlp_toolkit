"""
Text preprocess utilties
"""

import re
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

    #  Data Foramt by line:
        classification: token_1 token_2 ... token_n\tlabel
        sequence labeling: token_1 token_2 ... token_n\tlabel_1 label_2 ... label_n
    """

    def __init__(self, fname, task_type, max_words=80, segment=True):
        self.fname = fname
        self.task_type = task_type
        self.max_words = max_words
        self.transformer = IndexTransformer(self.max_words)
        self.segment = segment
        self.html_texts = re.compile(r'('+'|'.join(REGEX_STR)+')', re.UNICODE)
        self.load()

    def load(self):
        self.texts, self.labels = zip(
            *[line.strip().split('\t') for line in open(self.fname, 'r', encoding='utf8')])
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

    def transform(self, transformer_fname=''):
        cleaned_texts = (self.clean(line) for line in self.texts)
        x, y = zip(*[(x1.split(' '), y1.split(' ')) for x1, y1 in zip(cleaned_texts, self.labels) if x1])
        x, y = self.transformer.fit_transform(x, y)
        if transformer_fname:
            self.transformer.save(transformer_fname)
        logger.info('texts and labels transformed to number index')
        return x, y
