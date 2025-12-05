import math
import time

from eventvec.server.common.lists.said_verbs import said_verbs, past_perf_aux, pres_perf_aux, future_modals, future_said_verbs


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def token2tense(sentence, token):
    tense = None
    aspect = None
    advmod = None
    split_sentence = sentence.split()

    if token is None:
        return tense, aspect
    if token.pos() in ['VERB', 'ROOT', 'AUX']:
        #tense = 'Pres'
        if token.tense() is not None:
            tense = token.tense()
        aspect = token.aspect()
        aux_there = False
        if 'aux' in token.children():
            for child in token.children()['aux']:
                if child.text() == 'to':
                    if token.dep() in ['xcomp', 'advcl']:
                        tense, aspect, advmod = token2tense(sentence, token.parent())
                if child.tense() is not None:
                    tense = child.tense()
                if child.text() in past_perf_aux + pres_perf_aux:
                    aux_there = True
                    if aspect == 'Prog':
                        aspect = 'Prog-Perf'
                    else:
                        aspect = 'Perf'
        if aux_there is False and aspect == 'Perf':
            aspect = None
        if 'npadvmod' in token.children():
            for child in token.children()['npadvmod']:
                if child.text().lower() in ['today', 'tomorrow', 'everyday', 'yesterday', 'now']:
                    advmod = child.text().lower()
        
        if any(future_modal in ' '.join(split_sentence[max(0, token.i_in_sentence() - 4): token.i_in_sentence()]).lower() for future_modal in future_modals):
            tense = 'Future'
    return tense, aspect, advmod

def token2parent(token):
    deps = ['ccomp', 'xcomp', "parataxis", '-relcl', '-conj']
    parent = None
    use = False
    is_quote = None
    if token.dep() in deps:
        parent = token.parent()
        if (token.dep() in deps and (token.pos() in ['VERB', 'AUX']) or parent.dep() in ['ccomp', 'xcomp']):
            use = True
        while not (parent is None or parent.text() in said_verbs | future_said_verbs or parent.dep() == 'ROOT'):
            if (token.dep() in deps and (token.pos() in ['VERB', 'AUX']) or parent.dep() in ['ccomp', 'xcomp']):
                use = True
            parent = parent.parent()
    if use is False or (parent is not None and parent.text() not in said_verbs | future_said_verbs):
        parent = None
    if 'punct' in token.children():
            for child in token.children()['punct']:
                if child.text() in ["'", '"']:
                    is_quote = True
    return parent, is_quote


if __name__ == '__main__':
    from eventvec.server.featurizers.lingusitic_featurizer import LinguisticFeaturizer
    linguistic_featurizer = LinguisticFeaturizer()
    og_sentence = ('John has said Mary will have been going to school.')
    featurized_context = linguistic_featurizer.featurize_document(og_sentence)
    for sentence in featurized_context.sentences():
        for tokeni, token in enumerate(sentence.tokens()):
            if token.pos() == 'VERB' and sentence.tokens()[tokeni + 1] != 'VERB':
                print(token.text(), token2tense(og_sentence, token), token2parent(token))