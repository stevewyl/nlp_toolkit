import sys
sys.path.append('../')
from nlp_toolkit.data import Dataset
from nlp_toolkit.labeler import Labeler
import yaml

data_path = '../data/cv_word.txt'
config_path = '../config_sequence_labeling.yaml'

# 建议使用safe_load()
config = yaml.safe_load(open(config_path, encoding='utf8'))
config['data']['basic_token'] = 'char'
config['data']['use_seg'] = True
config['data']['use_radical'] = True

# 加载数据，初始化参数
dataset = Dataset(fname=data_path, task_type='sequence_labeling',
                  mode='train', config=config)

# 数据转换
x, y, new_config = dataset.transform()

# 定义标注器
seq_labeler = Labeler(config=new_config, model_name='char_rnn',
                      seq_type='bucket', transformer=dataset.transformer)

# 模型训练
# 会在当前目录生成models目录，用于保存模型训练结果
trained_model = seq_labeler.train(x, y)
