from collections import defaultdict
import csv

from eventvec.server.featurizers.lingusitic_featurizer import LinguisticFeaturizer
from eventvec.server.data.wikipedia.datahandlers.wiki_datahandler import WikiDatahandler


class ConnectorNLICreator():
    def __init__(self):
        self.wiki_datahandler = WikiDatahandler()
        self._counter = {'but': 0, 'and': 0, 'because': 0, 'so': 0, 'though': 0}
        self._linguistic_featurizer = LinguisticFeaturizer()
        self._dataset = defaultdict(list)

    def create_connector_dataset(self):
        wiki_datahandler = WikiDatahandler()
        for filename in wiki_datahandler.wiki_file_list():
            for line in wiki_datahandler.read_file(filename):

                featurized_doc = self._linguistic_featurizer.featurize_document(line)
                for sentence in featurized_doc.sentences():
                    for token in sentence.tokens():
                        if token.text() == 'but' and token.pos() == 'CCONJ':
                            self._counter['but'] += 1
                            if self._counter['but'] < 250:
                                self._dataset['but'].append(sentence.text())
                        if token.pos() in ['VERB', 'AUX'] and token.dep() in ['conj']:
                            parent = token.parent()
                            if 'cc' in parent.children():
                                children_text = [child.text() for child in parent.children()['cc']]
                                if 'and' in children_text:
                                    self._counter['and'] += 1
                                    if self._counter['and'] < 250:
                                        self._dataset['and'].append(sentence.text())
                        if token.pos() in ['SCONJ'] and token.dep() in ['mark'] and token.text() == 'so':
                            self._counter['so'] += 1
                            if self._counter['so'] < 250:
                                self._dataset['so'].append(sentence.text())
                        if token.pos() in ['SCONJ'] and token.dep() in ['mark'] and token.text() in ['because']:
                            self._counter['because'] += 1
                            if self._counter['because'] < 250:
                                self._dataset['because'].append(sentence.text())
                        if token.pos() in ['SCONJ'] and token.dep() in ['mark'] and token.text() in ['though']:
                            self._counter['though'] += 1
                            if self._counter['though'] < 250:
                                self._dataset['though'].append(sentence.text())
                        if token.pos() in ['ADV'] and token.dep() in ['advmod'] and token.text() in ['therefore']:
                            self._counter['so'] += 1
                            if self._counter['so'] < 250:
                                self._dataset['so'].append(sentence.text())
                print(self._counter)
                if all([value >= 250 for value in self._counter.values()]):
                    self.save_dataset()
                    return

    def save_dataset(self):
        with open('/home/lalady6977/oerc/projects/data/connector_nli.csv', 'wt') as f:
            writer = csv.writer(f, delimiter='\t')
            for key, value in self._dataset.items():
                for sentence in value:
                    writer.writerow([key, sentence])


if __name__ == '__main__':
    creator = ConnectorNLICreator()
    creator.create_connector_dataset()
    print(creator._counter)