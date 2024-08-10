import os
from typing import Dict
import json
from collections import defaultdict
import numpy as np
import csv
import numpy


from eventvec.server.data.mnli.chaos_mnli_data.chaos_mnli_datum import ChaosMNLIDatum
from eventvec.server.data.mnli.chaos_mnli_data.chaos_mnli_data import ChaosMNLIData
from eventvec.server.tasks.entailment_classification.featurizers.clause_matcher import ClauseMatcher
from eventvec.server.featurizers.factuality_categorizer.factuality_categorizer import FactualityCategorizer

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
    fr = ChaosMNLIDatareader()
    data = fr.chaos_data()
    divisions = [
        (0, 0.749),
        (0.749, 0.934),
        (0.934, 1.058),
        (1.058, 1.58),
    ]
    counter = defaultdict(int)
    fc = FactualityCategorizer()
    factuality_examples = defaultdict(list)
    non_factuality_examples = defaultdict(list)
    entropies = []
    cm = ClauseMatcher()
    all_entropies = []
    data = fr.chaos_data()
    examples = []
    interested = []
    for datum in sorted(data, key=lambda x: x.entropy(), reverse=True):
        if any(i in datum.premise().split() for i in ['but', 'because', 'so', 'therefore', 'however']) or any(i in datum.hypothesis() for i in ['but', 'because', 'so', 'therefore', 'however']):
            interested.append(datum)
        event_string, event_string_2 = cm.match(datum.premise(), datum.hypothesis())
        features1 = fc.categorize(datum.premise(), event_string).to_dict()
        features2 = fc.categorize(datum.hypothesis(), event_string_2).to_dict()
        if datum.majority_label() == 'c' and datum.entropy() > 1.4 and features1['is_subordinate_of_said'] is True:
            print(datum.premise(), '|', datum.hypothesis(), '|', datum.majority_label(), datum.entropy(), '\n' * 4)

        if (any (v is True for v in features1.values()) or any (v is True for v in features2.values())):
            for di, d in enumerate(divisions):
                if d[0] <= datum.entropy() < d[1]:
                    factuality_examples[di].append('{}|{}|{}'.format(datum.premise(), datum.hypothesis(), datum.majority_label()))
                    break
        else:
            for di, d in enumerate(divisions):
                if d[0] <= datum.entropy() < d[1]:
                    non_factuality_examples[di].append('{}|{}|{}'.format(datum.premise(), datum.hypothesis(), datum.majority_label()))
                    break

    total = len(data)
    for division in sorted(factuality_examples.keys()):
        print(division)
        total = len(factuality_examples[division]) + len(non_factuality_examples[division])
        print(division, len(factuality_examples[division]) / total, len(non_factuality_examples[division]) / total)
    print({i.uid() for i in interested})