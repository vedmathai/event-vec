import numpy as np
import re
from collections import defaultdict
from transformers import BertTokenizer


from eventvec.server.data_handlers.timebank_data_handler import TimeBankBertDataHandler  # noqa
from eventvec.server.data_handlers.bert_featurizer import BERTLinguisticFeaturizer  # noqa


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
    'none': 3,
}

labels_simpler_reverse = {
    'after': 0,
    'during': 1,
    'before': 2,
}

tenses = {
    'PRESPART': 0,
    'PASTPART': 1,
    'Pres': 2,
    'FUTURE': 3,
    'NONE': 4,
    'INFINITIVE': 5,
    'Past': 6,
    None: 4,
}

aspect = {
    'Perf': 0,
    'Prog': 1,
    'NONE': 2,
    'PERFECTIVE_PROGRESSIVE': 3,
    None: 2,
}

pos = {
    'VERB': 0,
    'X': 1,
    'AUX': 2,
    'ADJ': 3,
    'ADP': 4,
    'NOUN': 5,
    'ADV': 6,
    None: 7,
    'PROPN': 8,
}

tenses_hot_encoding = {i: [0]*7 for i in tenses}
for i in tenses_hot_encoding:
    tenses_hot_encoding[i][tenses[i]] = 1

aspect_hot_encoding = {i: [0]*4 for i in aspect}
for i in aspect_hot_encoding:
    aspect_hot_encoding[i][aspect[i]] = 1

pos_hot_encoding = {i: [0]*9 for i in pos}
for i in pos_hot_encoding:
    pos_hot_encoding[i][pos[i]] = 1


class BertDataHandler():
    def __init__(self):
        self._tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self._labels = set()
        self._label_counts = defaultdict(int)
        self._data_handler = TimeBankBertDataHandler()
        self._featurizer = BERTLinguisticFeaturizer()

    def load(self):
        self._model_input_data = self._data_handler.model_input_data()
        self._data_handler.load()
        self.load_train_data()
        self.load_test_data()

    def load_train_data(self):
        timebank_train_data = self._model_input_data.train_data()
        for datumi, datum in enumerate(timebank_train_data):
            self.process_timebank_data(datum)

    def load_test_data(self):
        timebank_test_data = self._model_input_data.test_data()
        for datumi, datum in enumerate(timebank_test_data):
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
        model_input_datum.set_from_decoded_sentence(decoded_sentence)
        decoded_sentence = self._tokenizer.batch_decode(whole_sentence_to_from['input_ids'])
        if 'entity1' not in decoded_sentence[0].split() or 'entity2' not in decoded_sentence[0].split():
            return
        to_from_from_token_i = decoded_sentence[0].split().index('entity1') + 1
        to_from_to_token_i = decoded_sentence[0].split().index('entity2') + 1
        model_input_datum.set_to_decoded_sentence(decoded_sentence)
        if any(i >= 200 for i in [from_to_from_token_i, from_to_to_token_i, to_from_from_token_i, to_from_to_token_i]):
            return
        label = labels_simpler[model_input_datum.relationship()]
        model_input_datum.set_is_trainable()
        model_input_datum.set_from_entity_token_i(from_to_from_token_i)
        model_input_datum.set_to_entity_token_i(from_to_to_token_i)
        model_input_datum.set_from_sentence_encoded(bert_encoding_from_sentence)  # noqa
        model_input_datum.set_to_sentence_encoded(bert_encoding_to_sentence)
        model_input_datum.set_sentence_pair_encoded(whole_sentence_from_to)
        model_input_datum.set_target(label)
        self._featurizer.featurize(model_input_datum)
        from_tense_encoding = tenses_hot_encoding[model_input_datum.from_tense()]
        to_tense_encoding = tenses_hot_encoding[model_input_datum.to_tense()]
        from_aspect_encoding = aspect_hot_encoding[model_input_datum.from_aspect()]
        to_aspect_encoding = aspect_hot_encoding[model_input_datum.to_aspect()]
        from_pos_encoding = pos_hot_encoding[model_input_datum.from_pos()]
        to_pos_encoding = pos_hot_encoding[model_input_datum.to_pos()]
        feature_encoding = [[model_input_datum.token_order()] + from_tense_encoding + to_tense_encoding + from_aspect_encoding + to_aspect_encoding + from_pos_encoding + to_pos_encoding]
        model_input_datum.set_feature_encoding(feature_encoding)
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
