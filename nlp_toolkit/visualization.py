"""
some Visualization Functions
"""
from seqeval.metrics.sequence_labeling import get_entities
from typing import List
import random

ENTITY_COLOR = ['#ff9900', '#00ccff', '#66ff99', '#ff3300', '#9933ff', '#669999']


def highlight_by_weight(word, att_weight):
    html_color = '#%02X%02X%02X' % (255, int(255 * (1 - att_weight)), int(255 * (1 - att_weight)))
    return '<span style="background-color: {}">{}</span>'.format(html_color, word)


def att2html(sentence, att_weights):
    html = ""
    for word, att_weight in zip(sentence.split(' '), att_weights):
        html += ' ' + highlight_by_weight(word, att_weight)
    return html + "<br><br>\n"


def attention_visualization(texts: List[str], attention_weights, lengths,
                            output_fname='attention_texts.html'):
    attention_true = [attention_weights[i][:lengths[i]]
                      for i in range(len(lengths))]
    with open(output_fname, 'w') as fout:
        for x, y in zip(texts, attention_true):
            fout.write(att2html(x, y))


def highlight_entity(words: List[str], entity_type, entity_color):
    if entity_type:
        html_color = entity_color[entity_type]
        words = ' '.join(words) + ' [%s]' % entity_type
        return '<span style="background-color: {}">{}</span>'.format(html_color, words)
    else:
        return ' '.join(words)


def entity2html(sentence, labels, entity_colors):
    html = ""
    entity_dict = {item[1]: [item[0], item[-1]] for item in labels}
    words = sentence.split(' ')
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


def entity_visualization(texts: List[str], labels: List[List[str]],
                         output_fname='entity_texts.html'):
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
        for x, y in zip(texts, entities):
            fout.write(entity2html(x, y, entity_colors))


def loss_acc_curve():
    pass
