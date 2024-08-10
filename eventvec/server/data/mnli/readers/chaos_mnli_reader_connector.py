import os
from typing import Dict
import json
from collections import defaultdict
import numpy as np
import csv
import numpy

from eventvec.server.featurizers.lingusitic_featurizer import LinguisticFeaturizer
from eventvec.server.data.mnli.chaos_mnli_data.chaos_mnli_datum import ChaosMNLIDatum
from eventvec.server.data.mnli.chaos_mnli_data.chaos_mnli_data import ChaosMNLIData
from eventvec.server.tasks.entailment_classification.featurizers.clause_matcher import ClauseMatcher
from eventvec.server.featurizers.factuality_categorizer.factuality_categorizer import FactualityCategorizer
from eventvec.server.data.mnli.mnli_datahandlers.mnli_data_reader import MNLIDataReader  # noqa
from eventvec.server.data.mnli.mnli_datahandlers.anli_data_reader import ANLIDataReader  # noqa



from eventvec.server.common.lists.said_verbs import said_verbs, future_said_verbs, future_modals
from eventvec.server.data.abstract import AbstractDatareader


class ChaosMNLIDatareader(AbstractDatareader):
    def __init__(self):
        super().__init__()
        self.folder = self._config.chaos_mnli_data_location()


    def read_file(self, filepath):
        with open(filepath) as f:
            return f.read()

    def chaos_data(self) -> Dict:
        filepath = self.folder
        data = []
        with open(filepath) as f:
            for line in f:
                json_data = json.loads(line)
                chaos_mnli_data = ChaosMNLIDatum.from_json(json_data)
                data.append(chaos_mnli_data)
        return data


if __name__ == '__main__':
    fr = ANLIDataReader()
    linguistic_featurizer = LinguisticFeaturizer()

    data = fr.read_file('test')
    interested = set()
    for datum in data.data():
        for text in [datum.sentence_1(), datum.sentence_2()]:
            featurized_doc = linguistic_featurizer.featurize_document(text)
            for sentence in featurized_doc.sentences():
                for token in sentence.tokens():
                    if token.text() == 'but' and token.pos() == 'CCONJ':
                        interested.add(datum.uid())
                    if token.pos() in ['VERB', 'AUX'] and token.dep() in ['conj']:
                        parent = token.parent()
                        if 'cc' in parent.children():
                            children_text = [child.text() for child in parent.children()['cc']]
                            if 'and' in children_text:
                                pass#interested.add(datum.uid())
                    if token.pos() in ['ADV'] and token.dep() in ['mark'] and token.text() == 'so':
                        interested.add(datum.uid())
                    if token.pos() in ['SCONJ'] and token.dep() in ['mark'] and token.text() in ['because']:
                        interested.add(datum.uid())
                    #if token.pos() in ['ADV'] and token.dep() in ['advmod'] and token.text() in ['therefore']:
                    #    interested.add(datum.uid())
    print(interested)
    print(len(interested))