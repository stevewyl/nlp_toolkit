def load_tc_data(fname, label_prefix='__label__', max_tokens_per_doc=-1):

    def gen():
        with open(fname, 'r', encoding='utf8') as fin:
            for line in fin:
                words = line.strip().split()
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
                    yield (text, label_line)

    texts, labels = zip(*[item for item in gen()])
    return texts, labels


def load_sl_data(fname, data_format='basic'):

    def process_conll(data):
        tokens, tags = [], []
        for line in data:
            if line:
                token, tag = line.split('\t')
                tokens.append(token)
                tags.append(tag)
            else:
                yield (tokens, tags)
                tokens, tags = [], []

    data = (line.strip() for line in open(fname, 'r', encoding='utf8'))
    if data_format:
        if data_format == 'basic':
            texts, labels = zip(
                *[zip(*[item.rsplit('###', 1) for item in line.split('\t')]) for line in data])
        elif data_format == 'conll':
            texts, labels = zip(*[item for item in process_conll(data)])
        return texts, labels
    else:
        print('invalid data format for sequence labeling task')