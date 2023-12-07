import json


class ChaosMNLIDatum:
    def __init__(self):
        self._uid = None
        self._premise = None
        self._hypothesis = None
        self._uid = None
        self._label_counter = None
        self._majority_label = None
        self._label_dist = None
        self._label_count = None
        self._entropy = None

    def set_uid(self, uid):
        self._uid = uid

    def uid(self):
        return self._uid

    def set_premise(self, premise):
        self._premise = premise

    def premise(self):
        return self._premise

    def set_hypothesis(self, hypothesis):
        self._hypothesis = hypothesis

    def hypothesis(self):
        return self._hypothesis

    def set_label_counter(self, label_counter):
        self._label_counter = label_counter

    def label_counter(self):
        return self._label_counter

    def set_majority_label(self, majority_label):
        self._majority_label = majority_label

    def majority_label(self):
        return self._majority_label

    def set_label_dist(self, label_dist):
        self._label_dist = label_dist

    def label_dist(self):
        return self._label_dist

    def set_label_count(self, label_count):
        self._label_count = label_count

    def label_count(self):
        return self._label_count

    def set_entropy(self, entropy):
        self._entropy = entropy

    def entropy(self):
        return self._entropy
    

    @staticmethod
    def from_json(jsonl):
        print(jsonl)
        datum = ChaosMNLIDatum()
        datum.set_uid(jsonl['uid'])
        datum.set_premise(jsonl['example']['premise'])
        datum.set_hypothesis(jsonl['example']['hypothesis'])
        datum.set_label_counter(jsonl['label_counter'])
        datum.set_majority_label(jsonl['majority_label'])
        datum.set_label_dist(jsonl['label_dist'])
        datum.set_label_count(jsonl['label_count'])
        datum.set_entropy(jsonl['entropy'])
        return datum
