import os
from typing import Dict
import json
from collections import defaultdict
import numpy as np
import csv

from eventvec.server.data.mnli.chaos_mnli_data.chaos_mnli_datum import ChaosMNLIDatum
from eventvec.server.data.mnli.chaos_mnli_data.chaos_mnli_data import ChaosMNLIData

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
        (0, 0.663),
        (0.663, 0.912),
        (0.912, 1.045),
        (1.045, 1.206),
        (1.206, 1.584),
    ]
    factuality_examples = defaultdict(list)
    non_factuality_examples = defaultdict(list)
    entropies = []
    factuality_entropies = defaultdict(int)
    non_factuality_entropies = defaultdict(int)
    check = set(future_modals) | set(future_said_verbs) | set(said_verbs)
    check = set(future_modals)
    for datum in sorted(data, key=lambda x: x.entropy(), reverse=True):
        if (any (i in datum.premise().split() for i in set(check)) or any (i in datum.hypothesis().split() for i in set(check))):
            for di, d in enumerate(divisions):
                if d[0] <= datum.entropy() < d[1]:
                    factuality_entropies[di] += 1
                    factuality_examples[di].append('{}|{}|{}'.format(datum.premise(), datum.hypothesis(), datum.majority_label()))
                    break
        else:
            for di, d in enumerate(divisions):
                if d[0] <= datum.entropy() < d[1]:
                    non_factuality_entropies[di] += 1
                    non_factuality_examples[di].append('{}|{}|{}'.format(datum.premise(), datum.hypothesis(), datum.majority_label()))
                    break
    total = sum(factuality_entropies.values())
    print([(i, factuality_entropies[i]/total) for i in range(0, 5)], total)
    for k in factuality_examples:
        for i in factuality_examples[k][0:5]:
            #print('factuality', k, i)
            #print()
            pass
    total = sum(non_factuality_entropies.values())
    print([(i, non_factuality_entropies[i]/total) for i in range(0, 5)], total)
    for k in non_factuality_examples:
        for i in non_factuality_examples[k][0:5]:
            # print('factuality', k, i)
            # print()
            pass
    print(total)
    print(len(data))