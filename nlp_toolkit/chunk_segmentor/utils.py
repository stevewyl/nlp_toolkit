"""一些nlp的常用函数"""

import re
import itertools
import collections
from hanziconv import HanziConv


# 扁平化列表
# ['1', '12', ['abc', 'df'], ['a']] ---> ['1','12','abc','df','a']
def flatten(x):
    tmp = [([i] if isinstance(i, str) else i) for i in x]
    return list(itertools.chain(*tmp))


def flatten_gen(x):
    for i in x:
        if isinstance(i, list) or isinstance(i, tuple):
            for inner_i in i:
                yield inner_i
        else:
            yield i


def n_grams(a, n):
    z = (itertools.islice(a, i, None) for i in range(n))
    return zip(*z)


def tag_by_dict(word_list, tree):
    idx = []
    length = len(word_list)
    start_idx = 0
    end_idx = 0
    while start_idx < length - 1:
        tmp_end_idx = 0
        tmp_chunk = ''.join(word_list[start_idx: end_idx+1])
        while tree.starts_with(tmp_chunk) and end_idx < length:
            tmp_end_idx = end_idx
            end_idx += 1
            tmp_chunk = ''.join(word_list[start_idx: end_idx+1])
        if tmp_end_idx != 0 and tree.search(''.join(word_list[start_idx: end_idx])):
            idx.append([start_idx, tmp_end_idx])
        start_idx += 1
        end_idx = start_idx
    if idx != []:
        idx = list(combine_idx(idx))
    return idx


# 合并交叉的chunk
def combine_idx(idx_list):
    l_idx = len(idx_list)
    if l_idx > 1:
        idx = 0
        used = []
        last_idx = l_idx - 1
        while idx <= l_idx - 2:
            if idx_list[idx+1][0] > idx_list[idx][1]:
                if idx not in used:
                    yield idx_list[idx]
                if idx + 1 == last_idx:
                    yield idx_list[idx+1]
                idx += 1
            else:
                start = idx_list[idx][0]
                while idx_list[idx+1][0] <= idx_list[idx][1]:
                    end = idx_list[idx+1][1]
                    used.append(idx)
                    idx += 1
                    if idx > l_idx - 2:
                        break
                used.append(idx)
                yield [start, end]
    else:
        yield idx_list[0]


def combine_two_idx(x, y):
    if x[0] >= y[0] and x[1] <= y[1]:
        return y
    elif x[0] < y[0] and x[1] > y[1]:
        return x
    else:
        all_idx = set(x + y)
        return [min(all_idx), max(all_idx)]


def compare_idx(dict_idx, model_idx):
    if dict_idx == model_idx or dict_idx == []:
        for idx in model_idx:
            yield idx
    elif model_idx == []:
        for idx in dict_idx:
            yield idx
    else:
        union_idx = dict_idx + model_idx
        uniq_idx = [list(x) for x in set([tuple(x) for x in union_idx])]
        sort_idx = sorted(uniq_idx, key=lambda x: (x[0], x[1]))
        for idx in list(combine_idx(sort_idx)):
            yield idx


def word_length(segs):
    cnt = []
    i = 0
    for item in segs:
        if item == 'E':
            i += 1
            cnt.append(i)
            i = 0
        elif item == 'S':
            cnt.append(1)
        else:
            i += 1
    return cnt


# 根据另外一个列表进行sub_list的切分
def split_sublist(list1, list2):
    if len(list1) == 1:
        return [list2]
    else:
        list1_len = [len(item) for item in list1]
        new_list = []
        for i in range(len(list1)):
            if i == 0:
                start = 0
            end = sum(list1_len[:i+1])
            new_list.append(list2[start: end])
            start = end
        return new_list


def output_reform(a, b, mode):
    if mode == 'accurate':
        return a + '_' + b
    else:
        return (a, b)


def reshape_term(term, qualifier_word=None, mode='accurate'):
    # pos = str(term.nature)
    # word = term.word
    term = str(term).split('/')
    pos = term[1]
    word = term[0]
    if pos == 'np':
        if word in qualifier_word:
            return output_reform(qualifier_word[word], pos, mode)
        else:
            return output_reform(word, pos, mode)
    else:
        return output_reform(word, pos, mode)


def hanlp_cut(sent_list, segmentor, qualifier_word=None, mode='accurate'):
    if qualifier_word is None:
        if mode == 'accurate':
            res = [[term.word + '_' + str(term.nature) for term in segmentor.segment(sub)] for sub in sent_list]
        else:
            res = [[(term.word, str(term.nature)) for term in segmentor.segment(sub)] for sub in sent_list]
    else:
        res = [[reshape_term(term, qualifier_word, mode) for term in segmentor.segment(sub)] for sub in sent_list]
    return res


def jieba_cut(sent_list, segmentor, qualifier_word=None, mode='accurate'):
    if qualifier_word is None:
        if mode == 'accurate':
            res = [[word + '_' + flag for word, flag in segmentor.cut(sub)] for sub in sent_list]
        else:
            res = [[(word, flag) for word, flag in segmentor.cut(sub)] for sub in sent_list]
    else:
        res = [[reshape_term(term, qualifier_word, mode) for term in segmentor.cut(sub)] for sub in sent_list]
    return res


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
START_PATTERN = r'(\d+、|\d+\.(?!\d+)|\d+\)|(?<![a-z0-9])[a-z]{1}(?=[、\.\)])|\(\d+\)|[一二三四五六七八九十]+[、\)\.)])'
END_PATTERN = r'(。|！|？|!|\?|；|;)'
HTML = re.compile(r'('+'|'.join(REGEX_STR)+')', re.UNICODE)


# 异常字符过滤
def preprocess(string):
    invalid_unicode = u'[\u25A0-\u25FF\u0080-\u00A0\uE000-\uFBFF\u2000-\u2027\u2030-\u206F]+'
    lang_char = u'[\u3040-\u309f\u30A0-\u30FF\u1100-\u11FF\u0E00-\u0E7F\u0600-\u06ff\u0750-\u077f\u0400-\u04ff]+'
    invalid_char = u'[\xa0\x7f\x9f]+'
    string = re.sub(EMOJI_UNICODE, '', string)
    string = re.sub(HTML, '', string)
    string = re.sub(r'\r|\t|<\w+>|&\w+;?|br\s*|li>', '', string)
    string = re.sub(invalid_char, '', string)
    string = re.sub(r'<U\+2028>|<U\+F09F>|<U\+F06C>|<U\+F0A7>', '', string)
    string = re.sub(r'[ \u3000]+', 's_', string)
    string = re.sub(invalid_unicode, 'ss_', string)
    string = re.sub(lang_char, 'lan_', string)
    # string = re.sub(r'(工作描述|工作职责|岗位职责|任职要求)(:|：)', '', string)
    string = HanziConv.toSimplified(strQ2B(string))
    string = re.sub(
        r'[^\u4e00-\u9fa5\u0020-\u007f，。！？；、（）：\n\u2029\u2028a-zA-Z0-9]+', '', string)
    return string


# 分句策略(比直接切开慢3倍)
def sent_split(string):
    string = re.sub(END_PATTERN, '\\1<cut>', re.sub(
        START_PATTERN, '<cut>\\1', string))
    return [item for item in re.split(r'\n|\u2029|\u2028|<cut>', string) if item != '']


def strQ2B(ustring):
    """全角转半角"""
    rstring = ""
    for uchar in ustring:
        inside_code = ord(uchar)
        if inside_code == 12288:
            inside_code = 32
        elif (inside_code >= 65281 and inside_code <= 65374):
            inside_code -= 65248
        rstring += chr(inside_code)
    return rstring.lower().strip()


'''
文件操作
'''


# 按行读取文本文件
def read_line(fname):
    return open(fname, encoding='utf8').read().split('\n')


# 保存为文本文件
def save_line(obj, fname='result.txt'):
    with open(fname, 'w', encoding='utf8') as f:
        if isinstance(obj, list):
            for k, v in enumerate(obj):
                v = str(v)
                if v != '\n' and k != len(obj) - 1:
                    f.write(v + '\n')
                else:
                    f.write(v)
        if isinstance(obj, collections.Counter) or isinstance(obj, dict):
            row = 0
            for k, v in sorted(obj.items(), key=lambda x: x[1], reverse=True):
                v = str(v)
                if str(v) != '\n' and k != len(obj) - 1:
                    f.write(k + '\t' + str(v) + '\n')
                    row += 1
                else:
                    f.write(k + '\t' + str(v))
