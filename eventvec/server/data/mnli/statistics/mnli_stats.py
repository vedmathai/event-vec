import os
from typing import Dict
import json
from collections import defaultdict
import numpy as np
import csv


from eventvec.server.common.lists.said_verbs import said_verbs, future_said_verbs, future_modals, negation_words, modal_adverbs, modal_adjectives
from eventvec.server.data.abstract import AbstractDatareader
from eventvec.server.data.mnli.mnli_datahandlers.mnli_data_reader import MNLIDataReader

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
    feature_stats = {}
    feature_set = modal_adjectives
    for word in feature_set:
        feature_stats[word] = FeatureStat()
        feature_stats[word].set_name(word)
    
    for datum in data.data():
        count = defaultdict(int)
        for label in datum.annotator_labels():
            count[label] += 1
        entropy = -sum([i/5 * np.log(i/5) for i in count.values()])
        for word in feature_set:
            spaced_word = f' {word} '
            if spaced_word in datum.sentence_1():
                feature_stats[word].add_example('{} | {}'.format(datum.sentence_1(), datum.sentence_2()))
                continue
            if spaced_word in datum.sentence_2():
                feature_stats[word].add_example('{} | {}'.format(datum.sentence_1(), datum.sentence_2()))

    for word in sorted(feature_set, key=lambda x: feature_stats[x].count(), reverse=False):
        print(f'{word}: {feature_stats[word].count()} / {len(data.data())}')
        for example in feature_stats[word].examples()[:10]:
            print(example)
        print('-----------------')
        print()