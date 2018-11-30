# ======主程序========
import sys
import os
import pickle
import time
import logging
from pathlib import Path
from collections import Counter

import jieba
import jieba.posseg as pseg
from nlp_toolkit.chunk_segmentor.trie import Trie
from nlp_toolkit.chunk_segmentor.utils import read_line, flatten_gen, sent_split, preprocess, jieba_cut
from nlp_toolkit.sequence import IndexTransformer
from nlp_toolkit.models import Word_RNN, Char_RNN
from nlp_toolkit.chunk_segmentor.tagger import Tagger

global model_loaded
global last_model_name
global Tree
global Labeler
global load_dict
global load_qualifier
global qualifier_dict
last_model_name = ''
tree_loaded = False
Labeler = None
Tree = None
load_dict = False
load_qualifier = False
qualifier_dict = None

# 关闭jieba的日志输出
jieba.setLogLevel(logging.INFO)


class Chunk_Labeler(object):
    def __init__(self, model_name='word-rnn', tagger=None):
        self.model_name = model_name
        if self.model_name != 'word-rnn':
            print('char-rnn model will update soon!')
            sys.exit()
        self.tagger = tagger

    def analyze(self, text, has_seq=True, char_input=False,
                mode='batch', batch_size=256, radical_file=''):
        if mode == 'single':
            batch_size = 1
        if not self.tagger:
            if self.model_name in ['char-rnn', 'idcnn']:
                char_input = True
            self.tagger = Tagger(self.model, self.p, char_input,
                                 mode, batch_size, radical_file)
        return self.tagger.analyze(text)

    @classmethod
    def load(cls, model_name, weight_file, params_file, preprocessor_file):
        self = cls(model_name=model_name)
        self.p = IndexTransformer.load(preprocessor_file)
        if model_name == 'word-rnn':
            self.model = Word_RNN.load(weight_file, params_file)
        elif model_name == 'char-rnn':
            self.model = Char_RNN.load(weight_file, params_file)
        else:
            print('No other available models for chunking')
            print('Please use word-rnn or char-rnn')
        return self


class Chunk_Segmentor(object):
    def __init__(self, user_dict='', model_name='word-rnn', mode='accurate', verbose=0):
        try:
            assert mode in ['accurate', 'fast']
        except:
            print('Only support three following mode: accurate, fast')
            sys.exit()
        self.pos = True
        self.mode = mode
        self.verbose = verbose
        self.path = os.path.abspath(os.path.dirname(__file__))
        if model_name != '':
            self.model_name = model_name
        else:
            try:
                self.model_name = read_line(Path(self.path) / 'data' / 'best_model.txt')[0]
            except Exception:
                self.model_name = model_name

        # jieba初始化
        base_dict = Path(self.path) / 'data' / 'dict' / 'jieba_base_supplyment.txt'
        jieba.load_userdict(str(base_dict))
        if mode == 'fast':
            global load_dict
            if not load_dict:
                dict_path = Path(self.path) / 'data' / 'dict' / 'chunk_pos.txt'
                jieba.load_userdict(str(dict_path))
                load_dict = True
        if user_dict:
            jieba.load_userdict(user_dict)
        self.seg = pseg

        # model变量
        self.weight_file = os.path.join(self.path, 'data/model/%s_weights.h5' % self.model_name)
        self.param_file = os.path.join(self.path, 'data/model/%s_parameters.json' % self.model_name)
        self.preprocess_file = os.path.join(self.path, 'data/model/%s_transformer.h5' % self.model_name)
        self.define_tagger()

    def define_tagger(self):
        global load_qualifier
        global qualifier_dict
        if not load_qualifier:
            qualifier_word_path = os.path.join(self.path, 'data/dict/chunk_qualifier.dict')
            self.qualifier_word = pickle.load(open(qualifier_word_path, 'rb'))
            load_qualifier = True
            qualifier_dict = self.qualifier_word
        else:
            self.qualifier_word = qualifier_dict

        self.basic_token = 'char' if self.model_name[:4] == 'char' else 'word'

        # acc模式变量
        if self.mode == 'accurate':
            global tree_loaded
            global last_model_name
            global Labeler
            global Tree
            if self.verbose:
                print('Model and Trie Tree are loading. It will cost 10-20s.')
            if self.model_name != last_model_name:
                self.labeler = Chunk_Labeler.load(
                    self.model_name, self.weight_file, self.param_file, self.preprocess_file)
                if self.verbose:
                    print('load model succeed')
                last_model_name = self.model_name
                Labeler = self.labeler
            else:
                self.labeler = Labeler
            if not tree_loaded:
                chunk_dict = read_line(os.path.join(self.path, 'data/dict/chunk.txt'))
                self.tree = Trie()
                for chunk in chunk_dict:
                    self.tree.insert(chunk)
                if self.verbose:
                    print('trie tree succeed')
                tree_loaded = True
                Tree = self.tree
            else:
                self.tree = Tree
            radical_file = os.path.join(self.path, 'data/dict/radical.txt')
            self.tagger = Tagger(self.labeler.model, self.labeler.p,
                                 basic_token=self.basic_token, radical_file=radical_file,
                                 tree=self.tree, qualifier_dict=self.qualifier_word)

    @property
    def get_segmentor_info(self):
        params = {'model_name': self.model_name,
                  'mode': self.mode,
                  'pos': self.pos}
        return params

    def extract_item(self, item):
        C_CUT_WORD, C_CUT_POS, C_CUT_CHUNK = 0, 1, 2
        complete_words = [sub[C_CUT_WORD] for sub in item]
        complete_poss = [sub[C_CUT_POS] for sub in item]
        if self.mode == 'fast':
            all_chunks = [x for sub in item for x, y in zip(sub[C_CUT_WORD], sub[C_CUT_POS]) if y == 'np']
        else:
            all_chunks = list(flatten_gen([sub[C_CUT_CHUNK] for sub in item]))
        words = list(flatten_gen(complete_words))
        poss = list(flatten_gen(complete_poss))
        if self.cut_all:
            words, poss = zip(*[(x1, y1) for x, y in zip(words, poss) for x1, y1 in self.cut_qualifier(x, y)])
        if self.pos:
            d = (words,   # C_CUT_WORD
                 poss,    # C_CUT_POS
                 list(dict.fromkeys(all_chunks)))   # C_CUT_CHUNK
        else:
            d = (words, list(dict.fromkeys(all_chunks)))
        return d

    def cut_qualifier(self, x, y):
        if y == 'np' and '_' in x:
            for sub_word in x.split('_'):
                yield sub_word, y
        else:
            yield x, y

    def output(self, data):
        idx_list, strings = zip(
            *[[idx, sub] for idx, item in enumerate(data) for sub in sent_split(preprocess(item))])
        cc = list(Counter(idx_list).values())
        end_idx = [sum(cc[:i]) for i in range(len(cc)+1)]
        seg_res = jieba_cut(strings, self.seg,
                            self.qualifier_word, mode=self.mode)
        if self.mode == 'accurate':
            outputs, _ = self.tagger.analyze(seg_res)
        else:
            outputs = [list(zip(*item)) for item in seg_res]
        new_res = (outputs[end_idx[i]: end_idx[i+1]]
                   for i in range(len(end_idx)-1))
        for item in new_res:
            yield self.extract_item(item)

    def cut(self, data, batch_size=512, pos=True, cut_all=False):
        if isinstance(data, str):
            data = [data]
        if not pos:
            self.pos = False
        else:
            self.pos = True
        if not cut_all:
            self.cut_all = False
        else:
            self.cut_all = True
        self.define_tagger()
        assert isinstance(data, list)
        data_cnt = len(data)
        num_batches = int(data_cnt / batch_size) + 1
        if self.verbose:
            print('total_batch_num: ', num_batches)
        for batch_num in range(num_batches):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_cnt)
            batch_input = data[start_index:end_index]
            for res in self.output(batch_input):
                yield res


if __name__ == "__main__":
    cutter = Chunk_Segmentor()
    cutter.cut('这是一个能够输出名词短语的分词器，欢迎试用！')
