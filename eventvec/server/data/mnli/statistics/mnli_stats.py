import os
from typing import Dict
import json
from collections import defaultdict
import numpy as np
import csv


from eventvec.server.common.lists.said_verbs import said_verbs, future_said_verbs, future_modals, negation_words, modal_adverbs, modal_adjectives, contrasting_conjunctions
from eventvec.server.data.abstract import AbstractDatareader
from eventvec.server.data.mnli.mnli_datahandlers.mnli_data_reader import MNLIDataReader
from eventvec.server.data.mnli.mnli_datahandlers.chaos_mnli_data_reader import ChaosMNLIDatareader
from eventvec.server.data.mnli.mnli_datahandlers.anli_data_reader import ANLIDataReader



class FeatureStat:
    def __init__(self):
        self._name = None
        self._examples = []

    def add_example(self, example):
        self._examples.append(example)

    def set_name(self, name):
        self._name = name

    def name (self):
        return self._name
    
    def count(self):
        return len(self._examples)
    
    def examples(self):
        return self._examples
    

if __name__ == '__main__':
    fr = MNLIDataReader()
    data = fr.read_file('train')
    count = 0
    contrasting_conjunctions = ['conversly']

    for datum in data.data()[:]:
        if any([word in datum.sentence_1().lower() for word in list(contrasting_conjunctions)]):
            print(datum.sentence_1())
            print()
        if any([word in datum.sentence_2().lower() for word in list(contrasting_conjunctions)]):
            print(datum.sentence_2())
            print()
    print(count)