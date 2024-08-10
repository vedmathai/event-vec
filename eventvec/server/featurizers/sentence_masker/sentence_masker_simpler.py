import re

from eventvec.server.featurizers.lingusitic_featurizer import LinguisticFeaturizer
from eventvec.server.common.lists.said_verbs import contrasting_conjunctions

other_conjuctions = {'and', 'because', 'but', 'so', 'or', 'though'}

conjuctions = contrasting_conjunctions | other_conjuctions
complement_deps = ['ccomp', 'xcomp']

auxilaries = ['is', "does", "can", "did", "will", 'do', 'are', 'was', 'were', 'have', 'has', 'had', 'am', 'be', 'being', 'been', 'could', 'would', 'should', 'may' ]


class SentenceMaskerSimpler:
    def __init__(self):
        self._linguistic_featurizer = LinguisticFeaturizer()
        self._fns = [
            self._but,
        ]

    def _but(self, fdoc):
        para = []
        i2change = {}
        i2before = {}
        mask_counter = 1
        required_answers = {}


        for sentence in fdoc.sentences():
            for token in sentence.tokens():
                if token.pos() in ['VERB', 'AUX']:
                    if 'cc' in token.children():
                        conjuction_children = token.children()['cc']
                        for conjunction in conjuction_children:
                            if conjunction.text() in other_conjuctions:
                                i2change[conjunction.i()] = ('MASK', conjunction.text())
                    if 'advmod' in token.children():
                        conjuction_children = token.children()['advmod']
                        for conjunction in conjuction_children:
                            if conjunction.text() in other_conjuctions:
                                i2change[conjunction.i()] = ('MASK', conjunction.text())
                    if 'mark' in token.children():
                        conjuction_children = token.children()['mark']
                        for conjunction in conjuction_children:
                            if conjunction.text() in other_conjuctions:
                                i2change[conjunction.i()] = ('MASK', conjunction.text())
                        
            new_sentence = []
            for token in sentence.tokens():
                if token.i() in i2before:
                    if i2before[token.i()][0] == 'MASK':
                        new_sentence.append('MASK_{}'.format(mask_counter))
                        required_answers['MASK_{}'.format(mask_counter)] = i2before[token.i()][1]
                        mask_counter += 1
                if token.i() in i2change:
                    
                    if i2change[token.i()][0] == 'MASK':
                        new_sentence.append('MASK_{}'.format(mask_counter))
                        required_answers['MASK_{}'.format(mask_counter)] = i2change[token.i()][1]
                        mask_counter += 1
                    elif i2change[token.i()][0][:4] != 'MASK':
                        new_sentence.append(i2change[token.i()][0])
                else:
                    new_sentence.append(token.text())

            para.extend(new_sentence)
            new_sentence = [i.strip() for i in new_sentence]
        return ' '.join(para), required_answers

    def mask(self, sentence):
        fdoc = self._linguistic_featurizer.featurize_document(sentence.lower())
        for fn in self._fns:
            return fn(fdoc)


if __name__ == '__main__':
    fc = SentenceMasker()
    t1 = fc.mask("Anita didn't know her real father, but her step father was around. Anita was mean to her stepfather.")
    print(t1)
