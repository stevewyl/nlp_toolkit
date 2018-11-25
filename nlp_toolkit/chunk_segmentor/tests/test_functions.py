import sys
# sys.path.append('../../..')
from nlp_toolkit.chunk_segmentor import Chunk_Segmentor
import time
import os

VERBOSE = 1
text = '主要配合UI设计师100%还原设计图，使用前端技术解决各大浏览器兼容问题，使用HTML5+css3完成页面优化及提高用户体验，www.s.com使用bootstrap、jQuery完成界面效果展示，使用JavaScript完成页面功能展示，并且，在规定的时间内提前完成任务大大提高了工作的效率'

print('test model loading')
cutter = Chunk_Segmentor(verbose=VERBOSE)

print('test Chunk_Segmentor object reload')
start = time.time()
cutter = Chunk_Segmentor(verbose=VERBOSE)
if time.time() - start < 1:
    pass
else:
    print('not pass reload model. Quit!')
    sys.exit()

'''
print('test switch model')
cutter = Chunk_Segmentor(model_name='char-rnn', verbose=VERBOSE)
print(list(cutter.cut(text)))
'''

print('test cutting performance')
cutter = Chunk_Segmentor(verbose=VERBOSE)
start = time.time()
print(list(cutter.cut(text, qualifier=False, pos=False)))
print('cut single sentence used {:04.2f}s'.format(time.time() - start))
print('test qualifier')
print(list(cutter.cut(text, qualifier=True)))
print('test pos')
print(list(cutter.cut(text, pos=True)))
print('test cut_all')
print(list(cutter.cut(text, cut_all=True)))

print('test user dict')
fin = open('user_dict.txt', 'w', encoding='utf8')
fin.write('用户体验 np\n')
fin.close()
cutter = Chunk_Segmentor(verbose=VERBOSE, user_dict='user_dict.txt')
print(list(cutter.cut(text)))
os.system('rm user_dict.txt')

text_list = [text] * 10000
start = time.time()
result = list(cutter.cut(text_list, qualifier=False, pos=False))
print('cut 10000 sentences no pos and quailifier used {:04.2f}s'.format(time.time() - start))
start = time.time()
result = list(cutter.cut(text_list))
print('cut 10000 sentences used {:04.2f}s'.format(time.time() - start))

print('test fast mode')
cutter = Chunk_Segmentor(mode='fast', verbose=VERBOSE)
print(list(cutter.cut(text)))
print('test cut_all')
print(list(cutter.cut(text, cut_all=True)))
start = time.time()
result = list(cutter.cut(text_list))
print('cut 10000 sentences in fast mode used {:04.2f}s'.format(time.time() - start))

print('test all pass')
