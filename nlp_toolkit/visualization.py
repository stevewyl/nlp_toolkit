"""
some Visualization Functions
"""


def highlight(word, att_weight):
    html_color = '#%02X%02X%02X' % (255, int(255*(1 - att_weight)), int(255*(1 - att_weight)))
    return '<span style="background-color: {}">{}</span>'.format(html_color, word)


def mk_html(sentence, att_weights):
    html = ""
    for word, att_weight in zip(sentence, att_weights):
        html += ' ' + highlight(word, att_weight)
    return html + "<br><br>\n"


def attention_visualization():
    pass


def loss_acc_curve():
    pass