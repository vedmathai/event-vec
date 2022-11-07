import numpy as np
import torch
import random
import re
from collections import defaultdict
from random import choices, shuffle
from transformers import BertTokenizer


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

labels_simpler_reverse = {
    'after': 0,
    'during': 1,
    'before': 2,
}


class BertDataHandler():

    def __init__(self):
        self._tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
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
        bert_encoding_from_sentence = self._tokenizer(
            [from_sentence],
            padding='max_length',
            max_length=200,
            truncation=True,
            return_tensors='pt',
        )
        bert_encoding_to_sentence = self._tokenizer(
            [to_sentence],
            padding='max_length',
            max_length=200,
            truncation=True,
            return_tensors='pt',
        )
        whole_sentence_from_to = self._tokenizer(
            [from_sentence], [to_sentence],
            padding='max_length',
            max_length=200,
            truncation=True,
            return_tensors='pt',
        )
        switched_from_sentence = re.sub('ENTITY1',  'ENTITY2', from_sentence)
        switched_to_sentence = re.sub('ENTITY2',  'ENTITY1', to_sentence)  
        whole_sentence_to_from = self._tokenizer(
            [switched_to_sentence], [switched_from_sentence],
            padding='max_length',
            max_length=200,
            truncation=True,
            return_tensors='pt',
        )
        decoded_sentence = self._tokenizer.batch_decode(whole_sentence_from_to['input_ids'])
        if 'entity1' not in decoded_sentence[0].split() or 'entity2' not in decoded_sentence[0].split():
            return
        from_to_from_token_i = decoded_sentence[0].split().index('entity1') + 1
        from_to_to_token_i = decoded_sentence[0].split().index('entity2') + 1
        decoded_sentence = self._tokenizer.batch_decode(whole_sentence_to_from['input_ids'])
        if 'entity1' not in decoded_sentence[0].split() or 'entity2' not in decoded_sentence[0].split():
            return
        to_from_from_token_i = decoded_sentence[0].split().index('entity1') + 1
        to_from_to_token_i = decoded_sentence[0].split().index('entity2') + 1
        if any(i >= 200 for i in [from_to_from_token_i, from_to_to_token_i, to_from_from_token_i, to_from_to_token_i]):
            return 
        feature_encoding = [datum['token_order']]
        feature_encoding_reverse = [1 - datum['token_order']]
        label = labels_simpler[datum['relationship']]
        label_reverse = labels_simpler_reverse[datum['relationship']]
        from_token_i = datum['from_token_i']
        to_token_i = datum['to_token_i']
        self._data.append((bert_encoding_from_sentence, bert_encoding_to_sentence, from_to_from_token_i, from_to_to_token_i, feature_encoding, from_sentence, to_sentence, whole_sentence_from_to, label))  # noqa
        #self._data.append((bert_encoding_to_sentence, bert_encoding_from_sentence, to_from_from_token_i, to_from_to_token_i, feature_encoding_reverse, switched_to_sentence , switched_from_sentence, whole_sentence_to_from, label_reverse))  # noqa
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
            weight = total / (float(self._label_counts[label]))
            weights[label] = weight
        return weights

    def data(self):
        return self._data

    def shuffle_data(self):
        random.shuffle(self._data)

    def sample_data(self, original_data):
        print(len(original_data))
        data = defaultdict(list)
        for datum in original_data:
            data[datum[-1]].append(datum)
        for datum in data:
            data[datum] = choices(data[datum], k=1000)
        data = sum([list(i) for i in data.values()], [])
        shuffle(data)
        return data

    def split_data(self):
        split_point_1 = int(.9*len(self._data))
        split_point_2 = int(1*len(self._data))
        split_data = np.split(self._data, [split_point_1, split_point_2])
        train_data, eval_data, test_data = split_data
        train_data = self.sample_data(train_data)
        return train_data, eval_data, test_data


if __name__ == '__main__':
    bh = BertDataHandler()
    bh.setup()
