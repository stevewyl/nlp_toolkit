"""
some Visualization Functions
"""
import random
from seqeval.metrics.sequence_labeling import get_entities
from typing import List
from copy import deepcopy

ENTITY_COLOR = ['#ff9900', '#00ccff', '#66ff99', '#ff3300', '#9933ff', '#669999']


def highlight_by_weight(word, att_weight):
    html_color = '#%02X%02X%02X' % (255, int(255 * (1 - att_weight)), int(255 * (1 - att_weight)))
    return '<span style="background-color: {}">{}</span>'.format(html_color, word)


def att2html(words, att_weights):
    html = ""
    for word, att_weight in zip(words, att_weights):
        html += ' ' + highlight_by_weight(word, att_weight)
    return html + "<br><br>\n"


def attention_visualization(texts: List[List[str]], attention_weights,
                            output_fname='attention_texts.html'):
    with open(output_fname, 'w') as fout:
        for x, y in zip(texts, attention_weights):
            fout.write(att2html(x, y))


def highlight_entity(words: List[str], entity_type, entity_color):
    if entity_type:
        html_color = entity_color[entity_type]
        words = ' '.join(words) + ' [%s]' % entity_type
        return '<span style="background-color: {}">{}</span>'.format(html_color, words)
    else:
        return ' '.join(words)


def entity2html(words, labels, entity_colors):
    html = ""
    entity_dict = {item[1]: [item[0], item[-1]] for item in labels}
    start, end = 0, 0
    while end < len(words):
        if end not in entity_dict:
            end += 1
            if end == len(words):
                html += words[-1]
        else:
            if end > start:
                html += highlight_entity(words[start: end], None, entity_colors) + ' '
            entity_info = entity_dict[end]
            entity_start = end
            entity_end = entity_info[-1] + 1
            html += highlight_entity(words[entity_start: entity_end], entity_info[0], entity_colors) + ' '
            start = entity_end
            end = start
    return html + "<br><br>\n"


def entity_visualization(texts: List[List[str]], labels: List[List[str]],
                         output_fname='entity_texts.html'):
    texts_c = deepcopy(texts)
    texts_c = [item[:-1] for item in texts_c]
    entities = [get_entities(item) for item in labels]
    all_entities = list(set([sub_item[0] for item in entities for sub_item in item]))
    all_entities = [item for item in all_entities if item != 'O']
    nb_entities = len(all_entities)
    if nb_entities > len(ENTITY_COLOR):
        rest_nb_colors = nb_entities - len(ENTITY_COLOR)
        colors = ENTITY_COLOR + ['#' + ''.join([random.choice('0123456789ABCDEF') for j in range(6)])
                                 for i in range(rest_nb_colors)]
    else:
        colors = ENTITY_COLOR[:nb_entities]
    assert len(colors) == nb_entities
    entity_colors = {all_entities[i]: colors[i] for i in range(nb_entities)}

    with open(output_fname, 'w') as fout:
        for x, y in zip(texts_c, entities):
            fout.write(entity2html(x, y, entity_colors))


def plot_loss_acc(history, task):
    import matplotlib.pyplot as plt

    nb_epochs = len(history.val_acc)
    epoch_size_nearly = len(history.acc) // nb_epochs
    val_x = [i for i in range(len(history.acc)) if i %
            epoch_size_nearly == 0][1:] + [len(history.acc)-1]

    f = plt.figure(figsize=(15, 45))
    ax1 = f.add_subplot(311)
    ax2 = f.add_subplot(312)
    ax3 = f.add_subplot(313)

    ax1.set_title("Train & Dev Acc")
    ax1.plot(history.acc, color="g", label="Train")
    ax1.plot(val_x, history.val_acc, color="b", label="Dev")
    ax1.legend(loc="best")

    ax2.set_title("Train & Dev Loss")
    ax2.plot(history.loss, color="g", label="Train")
    ax2.plot(val_x, history.val_loss, color="b", label="Dev")
    ax2.legend(loc="best")

    if task == 'classification':
        ax3.set_title("F1 per epoch")
        ax3.plot(history.metrics['f1'], color="g", label="F1")
    elif task == 'sequence_labeling':
        ax3.set_title("F1 and acc per epoch")
        ax3.plot(history.metrics['f1_seq'], color="g", label="F1")
        ax3.plot(history.metrics['seq_acc'], color="b", label="Acc")
    ax3.legend(loc="best")

    plt.tight_layout()
    plt.show()
