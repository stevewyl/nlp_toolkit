# Chunk分词器使用指南

环境依赖：python 3.6.5 (暂时只支持python3)

## 安装

```bash
pip install nlp_toolkit
```

## 主要功能

1. 能够输出名词短语
2. 支持词性输出，名词短语词性为np
3. 支持名词短语以限定词+中心词的形式输出

>不可分割的名词短语是不存在限定词+中心词的形式的，如“机器学习”，而“经典机器学习算法”可拆解为“经典_机器学习_算法”

## 如何使用

* 第一次import的时候，会自动下载模型和字典数据  
* 支持单句和多句文本的输入格式，建议以列表的形式传入分词器

```python
from nlp_toolkit.chunk_segmentor import Chunk_Segmentor
cutter = Chunk_Segmentor()
s = '这是一个能够输出名词短语的分词器，欢迎试用！'
res = [item for item in cutter.cut([s] * 10000)] # 1080ti上耗时8s

# 提供两个版本，accurate为精确版，fast为快速版但召回会降低一些，默认精确版
cutter = Chunk_Segmentor(mode='accurate')
cutter = Chunk_Segmentor(mode='fast')
# 支持用户自定义字典
# 格式为每行 词 [词性]，必须为utf8编码
cutter = Chunk_Segmentor(user_dict='your_dict.txt')
# 限定词+中心词的形式, 默认开启
cutter.cut(s, qualifier=False)
# 是否输出词性， 默认开启
cutter.cut(s, pos=False)
# 是否需要更细粒度的切分结果， 默认关闭
cutter.cut(s, cut_all=True)

# 输出格式（词列表，词性列表，名词短语集合）
[
    (
        ['这', '是', '一个', '能够', '输出', '名词_短语', '的', '分词器', ',', '欢迎', '试用', '!'],
        ['r', 'v', 'mq', 'v', 'vn', 'np', 'ude1', 'np', 'w', 'v', 'v', 'w'],
        ['分词器', '名词_短语']
    )
    ...
]
```

## Step 3 后续更新

若存在新的模型和字典数据，会提示你是否需要更新

## To-Do Lists

1. 提升限定词和名词短语的准确性 ---> 新的模型
2. char模型存在GPU调用内存溢出的问题 ---> 使用cnn提取Nchar信息来代替embedding的方式，缩小模型规模
3. 自定义字典，支持不同粒度的切分
4. 多进程模型加载和预测
