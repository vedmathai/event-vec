import numpy as np
import pprint
import re
from collections import defaultdict
from transformers import BertTokenizer, RobertaTokenizer
import pprint


from eventvec.server.data.timebank.datahandlers.timebank_data_handler import TimeBankBertDataHandler  # noqa
from eventvec.server.tasks.relationship_classification.featurizers.bert_featurizer import BERTLinguisticFeaturizer  # noqa


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
    'includes': 1,
    'is_included': 2,
    'during': 3,
    'after': 4,
    'none': 5,
    'modal': 5,
    'counter_factive': 5,
    'vague': 5,
    'factive': 5,
    'evidential': 5,
    'conditional': 5,
    'neg_evidential': 5,
    'factual': 5,
}

labels_simpler_reverse = {
    'after': 0,
    'during': 1,
    'before': 2,
}

tense_mapping = {
    "Pres": 0,
    "Past": 1,
    'Future': 2,
    None: 3,
}

aspect_mapping = {
    'Perf': 0,
    'Prog': 1,
    None: 2,
}

tenses_hot_encoding = {i: [0] * len(tense_mapping) for i in tense_mapping}
for i in tenses_hot_encoding:
    tenses_hot_encoding[i][tense_mapping[i]] = 1

aspect_hot_encoding = {i: [0] * len(aspect_mapping) for i in aspect_mapping}
for i in aspect_hot_encoding:
    aspect_hot_encoding[i][aspect_mapping[i]] = 1


class BertDataHandler():
    def __init__(self):
        self._tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
        self._labels = set()
        self._label_counts = defaultdict(int)
        self._data_handler = TimeBankBertDataHandler()
        self._featurizer = BERTLinguisticFeaturizer()
        self._aspect_counter = defaultdict(int)
        self._nouns = defaultdict(lambda: defaultdict(int))

    def load(self):
        self._model_input_data = self._data_handler.model_input_data()
        self._data_handler.load()
        self.load_train_data()
        self.load_test_data()

    def load_train_data(self):
        timebank_train_data = self._model_input_data.train_data()
        for datumi, datum in enumerate(timebank_train_data):
            self.process_timebank_data(datum, 'train')

    def load_test_data(self):
        timebank_test_data = self._model_input_data.test_data()
        for datumi, datum in enumerate(timebank_test_data):
            self.process_timebank_data(datum, 'test')
        train_set = set(self._nouns['train'].keys())
        test_set = set(self._nouns['test'].keys())
        num = 0
        den = 0
        for i in train_set & test_set:
            num += self._nouns['test'][i]
        for i in test_set:
            den += self._nouns['test'][i]
        #print(num/den)
        print(self._label_counts)

    def process_timebank_data(self, model_input_datum, is_train_or_test):
        from_sentence = ' '.join(model_input_datum.from_sentence())
        to_sentence = ' '.join(model_input_datum.to_sentence())
        bert_encoding_from_sentence = self._tokenizer(
            [from_sentence],
            padding='max_length',
            max_length=200,
            truncation=True,
            return_tensors='pt',
            return_token_type_ids=True
        )
        bert_encoding_to_sentence = self._tokenizer(
            [to_sentence],
            padding='max_length',
            max_length=200,
            truncation=True,
            return_tensors='pt',
            return_token_type_ids=True
        )
        whole_sentence_from_to = self._tokenizer(
            [from_sentence], [to_sentence],
            padding='max_length',
            max_length=200,
            truncation=True,
            return_tensors='pt',
            return_token_type_ids=True
        )
        switched_from_sentence = re.sub('entity1',  'entity2', from_sentence)
        switched_to_sentence = re.sub('entity2',  'entity1', to_sentence)
        whole_sentence_to_from = self._tokenizer(
            [switched_to_sentence], [switched_from_sentence],
            padding='max_length',
            max_length=200,
            truncation=True,
            return_tensors='pt',
            return_token_type_ids=True
        )
        decoded_sentence = self._tokenizer.batch_decode(whole_sentence_from_to['input_ids'])
        if 'entity1' not in decoded_sentence[0].split() or 'entity2' not in decoded_sentence[0].split():
            return
        from_to_from_token_i = decoded_sentence[0].split().index('entity1') + 1
        from_to_to_token_i = decoded_sentence[0].split().index('entity2') + 1
        model_input_datum.set_from_decoded_sentence(decoded_sentence)
        #decoded_sentence = self._tokenizer.batch_decode(whole_sentence_to_from['input_ids'])
        if 'entity1' not in decoded_sentence[0].split() or 'entity2' not in decoded_sentence[0].split():
            return
        to_from_from_token_i = decoded_sentence[0].split().index('entity1') + 1
        to_from_to_token_i = decoded_sentence[0].split().index('entity2') + 1
        model_input_datum.set_to_decoded_sentence(decoded_sentence)
        #if any(i >= 200 for i in [from_to_from_token_i, from_to_to_token_i, to_from_from_token_i, to_from_to_token_i]):
        #    return
        label = labels_simpler[model_input_datum.relationship()]
        model_input_datum.set_is_trainable(True)
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
        """
        from_sentence = ' '.join(model_input_datum.marked_up_parent_from_sentence())
        to_sentence = ' '.join(model_input_datum.marked_up_parent_to_sentence())
        whole_sentence_from_to = self._tokenizer(
            [from_sentence], [to_sentence],
            padding='max_length',
            max_length=200,
            truncation=True,
            return_tensors='pt',
            return_token_type_ids=True
        )
        decoded_sentence = self._tokenizer.batch_decode(whole_sentence_from_to['input_ids'])
        #if 'entity1' not in decoded_sentence[0].split() or 'entity2' not in decoded_sentence[0].split():
        #    return
        from_to_from_token_i = decoded_sentence[0].split().index('entity1') + 1
        from_to_to_token_i = decoded_sentence[0].split().index('entity2') + 1
        model_input_datum.set_from_entity_token_i(from_to_from_token_i)
        model_input_datum.set_to_entity_token_i(from_to_to_token_i)
        model_input_datum.set_sentence_pair_encoded(whole_sentence_from_to)
        """
        feature_encoding = [from_tense_encoding + from_aspect_encoding + to_tense_encoding + to_aspect_encoding]
        model_input_datum.set_feature_encoding(feature_encoding)
        from_token = decoded_sentence[0].split()[from_to_from_token_i]
        to_token = decoded_sentence[0].split()[from_to_to_token_i]
        from_to_tokens = '{}|{}'.format(from_token, to_token) + str(label)
        if is_train_or_test == 'train':
            model_input_datum.set_is_trainable(True)
        if is_train_or_test == 'test':
            model_input_datum.set_is_trainable(True)
        self._model_input_data.add_class(label)
        self._labels.add(label)


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
