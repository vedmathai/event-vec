from collections import defaultdict
import json
import numpy as np

from eventvec.server.data_readers.torque_reader.torque_datareader import TorqueDataReader
from eventvec.server.featurizers.lingusitic_featurizer import LinguisticFeaturizer

with open('eventvec/server/data_handlers/book_corpus_datahandlers/book_corpus_noun_events.json') as f:
    noun_events = set(json.load(f))

prep_set = set(['after', 'during', 'before', "between", "by", "following", "for", "from", "on", "since", "till", "to", "until", "within", "while", "except"])


if __name__ == '__main__':
    tdr = TorqueDataReader()
    ds = tdr.torque_dataset()
    counter = defaultdict(int)
    acc = []

    for d in ds[0].data():
        nums = 0
        dens = 0
        events = []
        linguistic_featurizer = LinguisticFeaturizer()
        featurized = linguistic_featurizer.featurize_document(d.passage())  # noqa
        tokeni2token = {}
        for sentence in featurized.sentences():
            for token in sentence.tokens():
                if token.pos() in ['VERB', 'AUX']:

                    if token.tense() == 'Past':
                        if token.pos() == 'VERB':
                            if token.parent() is not None and token.parent().pos() not in ['AUX']:
                                events.append(token.text())
                            elif token.parent() is None:
                                events.append(token.text())
                        if token.pos() == 'AUX' and token.parent() is not None and token.parent().pos() not in ['VERB']:
                            events.append(token.text())
                    if token.tense() == 'Pres':
                        children = token.all_children()
                        children = [i.text() for i in children]
                        if any(i in children for i in ['had', 'has']):
                            events.append(token.text())
                if token.pos() == 'NOUN' and token.dep() == 'pobj' and token.text().lower() in noun_events and token.parent().text() in prep_set:
                    events.append(token.text())

        
        for question in d.question_answer_pairs().questions():
            if question.question() == 'What event has already happened?':
                past_events = question.answer().spans()
                print(d.passage())
                print(' ')
                print(question.question())
                print('pred_events', events)
                print('label_events', past_events)
                print('-'*48)
        
        if len(set(events)) > 0:
            nums += len(set(events) & set(past_events))
            dens += len(set(events))
            acc += [nums / dens]
    print(np.mean(acc))

    """
    for indice in d.events().answer().indices():
        if indice[0] in tokeni2token:
            word = tokeni2token[indice[0]]
            dist = len(word.text()) - indice[1] + indice[0]
            if dist == 0:
                if word.pos() in ["NOUN"]:
                    verbs.append(word.text())
                    #print(d.passage()[:indice[0]] + '------->' + d.passage()[indice[0]: indice[1]] + '<--------' + d.passage()[indice[1]:])

    print(all_words)
    """