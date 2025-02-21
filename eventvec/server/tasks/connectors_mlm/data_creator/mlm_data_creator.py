from collections import defaultdict
import csv

from eventvec.server.featurizers.lingusitic_featurizer import LinguisticFeaturizer
from eventvec.server.data.wikipedia.datahandlers.wiki_datahandler import WikiDatahandler

from eventvec.server.data.roc_stories.datahandlers.roc_stories_datareader import ROCStoriesDatareader


class ConnectorMLMCreator():
    def __init__(self):
        self._wiki_datahandler = WikiDatahandler()
        self._roc_datareader = ROCStoriesDatareader()
        self._counter = {'but': 0, 'and': 0, 'because': 0, 'so': 0, 'though': 0}
        self._linguistic_featurizer = LinguisticFeaturizer()
        self._dataset = defaultdict(list)

    def create_wiki_connector_dataset(self):
        lines = []
        wiki_datahandler = WikiDatahandler()
        for filename in wiki_datahandler.wiki_file_list():
            for line in wiki_datahandler.read_file(filename):
                lines.append(line)
    
    def create_roc_connector_dataset(self):
        data = self._roc_datareader.read_file()
        total = len(data)
        for datum_i, datum in enumerate(data):
            if datum_i % 100 == 0:
                print(datum_i, total)
            text = datum.text()
            self.search_words(text)
  
    def search_words(self, text):
        if any(i in text.strip() for i in ['and', 'but', 'because', 'so', 'though']):
            featurized_doc = self._linguistic_featurizer.featurize_document(text)
            for sentence in featurized_doc.sentences():
                for token in sentence.tokens():
                    if token.text() == 'but' and token.pos() == 'CCONJ':
                        self._counter['but'] += 1
                        if self._counter['but'] < 500:
                            token.set_text('[MASK]')
                            self._dataset['but'].append(self.get_sentences(featurized_doc))
                            token.set_text('but')
                    if token.pos() in ['VERB', 'AUX'] and token.dep() in ['conj']:
                        parent = token.parent()
                        if 'cc' in parent.children():
                            for child in parent.children()['cc']:
                                if child.text() == 'and':
                                    self._counter['and'] += 1
                                    if self._counter['and'] < 500:
                                        child.set_text('[MASK]')
                                        self._dataset['and'].append(self.get_sentences(featurized_doc))
                                        child.set_text('and')
                    if token.pos() in ['SCONJ'] and token.dep() in ['mark'] and token.text() == 'so':
                        self._counter['so'] += 1
                        if self._counter['so'] < 500:
                            token.set_text('[MASK]')
                            self._dataset['so'].append(self.get_sentences(featurized_doc))
                            token.set_text('so')
                    if token.pos() in ['SCONJ'] and token.dep() in ['mark'] and token.text() in ['because']:
                        self._counter['because'] += 1
                        if self._counter['because'] < 500:
                            token.set_text('[MASK]')
                            self._dataset['because'].append(self.get_sentences(featurized_doc))
                            token.set_text('because')
                    if token.pos() in ['SCONJ'] and token.dep() in ['mark'] and token.text() in ['though']:
                        changed = ''
                        self._counter['though'] += 1
                        if self._counter['though'] < 500:
                            token.set_text('[MASK]')
                            if sentence.tokens()[-1].text().strip().lower() == 'even':
                                sentence.tokens()[-1].set_text('')
                                changed = sentence.tokens()[-1].text()
                            self._dataset['though'].append(self.get_sentences(featurized_doc))
                            token.set_text('though')
                            if changed != '':
                                sentence.tokens()[-1].set_text(changed)

                    if token.pos() in ['ADV'] and token.dep() in ['advmod'] and token.text() in ['therefore']:
                        self._counter['so'] += 1
                        if self._counter['so'] < 500:
                            token.set_text('[MASK]')
                            self._dataset['so'].append(self.get_sentences(featurized_doc))
                            token.set_text('so')
        print(self._counter)
        self.save_dataset()
        
    def get_sentences(self, doc):
        text = []
        for sentence in doc.sentences():
            text.append(sentence.text())
        return ' '.join(text)

    def save_dataset(self):
        with open('/home/lalady6977/oerc/projects/data/roc_masked_connectors.csv', 'wt') as f:
            writer = csv.writer(f, delimiter='\t')
            for key, value in self._dataset.items():
                for sentence in value:
                    writer.writerow([key, sentence])


if __name__ == '__main__':
    creator = ConnectorMLMCreator()
    creator.create_roc_connector_dataset()
    print(creator._counter)