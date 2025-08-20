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
    'neutral': 'neutral',
    'entailment': 'entailment',
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
        train_size = 10000
        data_reader = self._data_readers['mnli']
        #chaos_data_reader = self._chaos_data_readers[run_config.dataset()]
        data = data_reader.read_file('train').data()
        random.seed(42)
        random.shuffle(data)
        mnli_data = []
        cnli_data = []
        self._train_data = []
        self._test_data = []
        connectors = ['and', 'though', 'but', 'because', 'so', 'therefore']
        for datum in data:
            if (any(i in datum.sentence_1().split() for i in connectors) or any(i in datum.sentence_2().split() for i in connectors)):
                mnli_data.append(datum)
        mnli_data.extend(random.sample(data, 5000))
        random.shuffle(mnli_data)
        data_reader = self._data_readers['cnli']
        cnli_data = data_reader.read_file('train').data()
        random.seed(42)
        random.shuffle(cnli_data)
        if 'mnli' in run_config.dataset():
            mnli_data_train = mnli_data[:int(0.8 * len(mnli_data))][:10000]
            self._train_data.extend(mnli_data_train)
        if 'mnli' in run_config.test_dataset():
            self._test_data.extend(mnli_data[int(0.8 * len(mnli_data)):][:2000])
        if 'cnli' in run_config.dataset():
            self._train_data.extend(cnli_data[:int(0.8 * len(cnli_data))])
        if 'cnli' in run_config.test_dataset():
            self._test_data.extend(cnli_data[int(0.8 * len(cnli_data)):])
        
        del(mnli_data_train)
        del(mnli_data)

        random.shuffle(self._train_data)
        random.shuffle(self._test_data)
        print(len(self._train_data))
        print(len(self._test_data))

        #data = chaos_data_reader.read_file('test').data()
        for datum in self._train_data:
            datum.set_label(label_map[datum.label()])
        for datum in self._test_data:
            datum.set_label(label_map[datum.label()])

    def train_data(self):
        return self._train_data

    def test_data(self):
        return self._test_data
