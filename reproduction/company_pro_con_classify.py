from nlp_toolkit.data import Dataset
from nlp_toolkit.classifier import Classifier
import yaml

data_path = '../sample_data/company_pro_con.txt'
config_path = '../config_classification.yaml'

# 建议使用safe_load()
config = yaml.safe_load(open(config_path, encoding='utf8'))
config['model']['bi_lstm_att']['return_attention'] = True

# 加载数据，初始化参数
dataset = Dataset(fname=data_path, task_type='classification',
                  mode='train', config=config)

# 定义分类器
classifier = Classifier(model_name='bi_lstm_att', dataset=dataset,
                        seq_type='bucket')

# 模型训练
# 会在当前目录生成models目录，用于保存模型训练结果
trained_model = classifier.train()
