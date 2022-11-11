import numpy as np
import re
from collections import defaultdict
from transformers import BertTokenizer



from eventvec.server.data_handlers.model_input.model_input_data import ModelInputData  # noqa
from eventvec.server.data_handlers.model_input.model_input_datum import ModelInputDatum  # noqa
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
    'identity': 3,
}

labels_simpler_reverse = {
    'after': 0,
    'during': 1,
    'before': 2,
}

tenses = {
    'PRESPART': 0,
    'PASTPART': 1,
    'PRESENT': 2,
    'FUTURE': 3,
    'NONE': 4,
    'INFINITIVE': 5,
    'PAST': 6
}

tenses_hot_encoding = {i: [0]*7 for i in tenses}
for i in tenses_hot_encoding:
    tenses_hot_encoding[i][tenses[i]] = 1


class BertDataHandler():

    def __init__(self):
        self._tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self._labels = set()
        self._label_counts = defaultdict(int)
        self._data_handler = TimeBankBertDataHandler()

    def load(self):
        self._model_input_data = self._data_handler.model_input_data()
        self._data_handler.load()
        self.load_train_data()
        self.load_test_data()

    def load_train_data(self):
        timebank_train_data = self._model_input_data.train_data()
        for datumi, datum in enumerate(timebank_train_data):
            if datumi < 100:
                self.process_timebank_data(datum)

    def load_test_data(self):
        timebank_test_data = self._model_input_data.test_data()
        for datumi, datum in enumerate(timebank_test_data):
            if datumi < 100:
                self.process_timebank_data(datum)

    def process_timebank_data(self, model_input_datum):
        from_sentence = ' '.join(model_input_datum.from_sentence())
        to_sentence = ' '.join(model_input_datum.to_sentence())
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
        from_tense_encoding = tenses_hot_encoding[model_input_datum.from_tense()]
        to_tense_encoding = tenses_hot_encoding[model_input_datum.to_tense()]
        feature_encoding = [[model_input_datum.token_order()] + from_tense_encoding + to_tense_encoding]
        label = labels_simpler[model_input_datum.relationship()]
        model_input_datum.set_is_trainable()
        model_input_datum.set_from_entity_token_i(from_to_from_token_i)
        model_input_datum.set_to_entity_token_i(from_to_to_token_i)
        model_input_datum.set_from_sentence_encoded(bert_encoding_from_sentence)  # noqa
        model_input_datum.set_to_sentence_encoded(bert_encoding_to_sentence)
        model_input_datum.set_feature_encoding(feature_encoding)
        model_input_datum.set_sentence_pair_encoded(whole_sentence_from_to)
        model_input_datum.set_target(label)
        self._model_input_data.add_class(label)
        self._labels.add(label)
        self._label_counts[label] += 1

    def model_input_data(self):
        return self._model_input_data

    def labels(self):
        return self._labels

    def label_counts(self):
        return dict(self._label_counts)

    def label_weights(self):
        weights = [0 for i in range(len(self.labels()))]
        total = sum(self._label_counts.values())
        for label in self.labels():
            weight = total / (float(self._label_counts[label]))
            weights[label] = weight
        return weights
