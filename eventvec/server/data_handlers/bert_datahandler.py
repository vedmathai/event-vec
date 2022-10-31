import numpy as np
import torch
from transformers import BertTokenizer
import random
from collections import defaultdict


from eventvec.server.data_handlers.timebank_data_handler import TimeBankBertDataHandler  # noqa


labels = {
    'is_included': 0,
    'simultaneous': 1,
    'before': 2,
    'identity': 3,
    'during': 4,
    'ended_by': 5,
    'begun_by': 6,
    'i_after': 7,
    'after': 8,
    'i_before': 9,
    'ends': 10,
    'includes': 11,
    'during_inv': 12,
    'begins': 13,
}

labels_simpler = {
    'before': 0,
    'during': 1,
    'after': 2,
}


class BertDataHandler():

    def __init__(self):
        self._tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
        self._data = []
        self._labels = []
        self._label_counts = defaultdict(int)

    def load(self):
        data_handler = TimeBankBertDataHandler()
        timebank_data = data_handler.get_data()
        for datum in timebank_data:
            self.process_timebank_data(datum)
        self.shuffle_data()

    def process_timebank_data(self, datum):
        from_sentence = ' '.join(datum['from_sentence'])
        to_sentence = ' '.join(datum['to_sentence'])
        encoding = self._tokenizer(
            [from_sentence], [to_sentence],
            padding='max_length',
            max_length=200,
            truncation=True,
            return_tensors='pt',
        )
        label = labels_simpler[datum['relationship']]
        self._data.append((encoding, label))
        self._labels.append(label)
        self._label_counts[label] += 1

    def classes(self):
        return set(self._labels)

    def label_counts(self):
        return dict(self._label_counts)

    def label_weights(self):
        weights = [0 for i in range(len(self.classes()))]
        total = sum(self._label_counts.values())
        for label in self.classes():
            weight = float(total - self._label_counts[label]) / total
            weights[label] = weight
        return weights

    def data(self):
        return self._data

    def shuffle_data(self):
        random.shuffle(self._data)

    def split_data(self):
        split_point_1 = int(.9*len(self._data))
        split_point_2 = int(1*len(self._data))
        split_data = np.split(self._data, [split_point_1, split_point_2])
        return split_data


if __name__ == '__main__':
    bh = BertDataHandler()
    bh.setup()
