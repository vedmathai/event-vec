import numpy as np
import pprint
import re
from collections import defaultdict
from transformers import BertTokenizer, RobertaTokenizer
import pprint
import random
from collections import defaultdict


from eventvec.server.data.timebank.datahandlers.timebank_data_handler import TimeBankBertDataHandler  # noqa
from eventvec.server.tasks.relationship_classification.featurizers.bert_featurizer import BERTLinguisticFeaturizer  # noqa

from eventvec.server.data.mnli.mnli_datahandlers.mnli_data_reader import MNLIDataReader  # noqa
from eventvec.server.data.mnli.mnli_datahandlers.mnli_syntax_data_reader import MNLISyntaxDataReader  # noqa
from eventvec.server.data.mnli.mnli_datahandlers.snli_data_reader import SNLIDataReader  # noqa
from eventvec.server.data.mnli.mnli_datahandlers.anli_data_reader import ANLIDataReader  # noqa

from eventvec.server.data.mnli.mnli_datahandlers.chaos_mnli_syntax_data_reader import ChaosMNLISyntaxDataReader  # noqa
from eventvec.server.data.mnli.mnli_datahandlers.chaos_mnli_data_reader import ChaosMNLIDatareader  # noqa
from eventvec.server.data.mnli.mnli_datahandlers.chaos_snli_data_reader import ChaosSNLIDatareader  # noqa
from eventvec.server.data.mnli.mnli_datahandlers.chaos_anli_data_reader import ChaosANLIDatareader  # noqa
from eventvec.server.data.mnli.mnli_datahandlers.connector_nli_data_reader import ConnectorNLIDatareader

label_map = {
    'contradiction': 'contradiction',
    'non-strict': 'entailment',
    'strict': 'entailment',
}


class NLIDataHandler():
    def __init__(self):
        self._data_readers = {
            'mnli': MNLIDataReader(),
            'mnli_syntax': MNLISyntaxDataReader(),
            'snli': SNLIDataReader(),
            'anli': ANLIDataReader(),
            'cnli': ConnectorNLIDatareader(),
        } 

        self._chaos_data_readers = {
            'mnli': ChaosMNLIDatareader(),
            'mnli_syntax': ChaosMNLISyntaxDataReader(),
            'snli': ChaosSNLIDatareader(),
            'anli': ChaosANLIDatareader(),
        }

    def load(self, run_config):
        data_reader = self._data_readers[run_config.dataset()]
        #chaos_data_reader = self._chaos_data_readers[run_config.dataset()]
        data = data_reader.read_file('train').data()
        random.seed(42)
        random.shuffle(data)
        train_size = 10000
        train_data = []
        #connectors = ['and', 'though', 'but', 'because', 'so', 'therefore']
        #self._train_data = data#[:train_size]
        #for datum in self._train_data:
        #   if (any(i in datum.sentence_1().split() for i in connectors) or any(i in datum.sentence_2().split() for i in connectors)):
        #        train_data.append(datum)
        #train_data.extend(random.sample(self._train_data, 5000))
        train_data = data[:int(.8*len(data))]
        test_data = data[int(.8*len(data)):]
        self._train_data = train_data[:10000]
        #random.shuffle(self._train_data)
        data_reader = self._data_readers[run_config.test_dataset()]
        test_data = data_reader.read_file('test').data()
        #data = chaos_data_reader.read_file('test').data()
        self._test_data = test_data
        for datum in self._train_data:
            datum.set_label(label_map[datum.label()])
        for datum in self._test_data:
            datum.set_label(label_map[datum.label()])

    def train_data(self):
        return self._train_data

    def test_data(self):
        return self._test_data
