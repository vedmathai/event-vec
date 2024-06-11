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



class NLIDataHandler():
    def __init__(self):
        self._data_readers = {
            'mnli': MNLIDataReader(),
            'mnli_syntax': MNLISyntaxDataReader(),
            'snli': SNLIDataReader(),
            'anli': ANLIDataReader(),
        } 

        self._chaos_data_readers = {
            'mnli': ChaosMNLIDatareader(),
            'mnli_syntax': ChaosMNLISyntaxDataReader(),
            'snli': ChaosSNLIDatareader(),
            'anli': ChaosANLIDatareader(),
        }

    def load(self, run_config):
        data_reader = self._data_readers[run_config.dataset()]
        chaos_data_reader = self._chaos_data_readers[run_config.dataset()]
        data = data_reader.read_file('train').data()
        random.seed(42)
        random.shuffle(data)
        train_size = 10000
        self._train_data = data[:train_size]
        #for i in ['contradiction' , 'neutral', 'entailment']:
        #    self._train_data.extend(list(filter(lambda x: x.label() == i, data))[:int(train_size/3)])
        random.shuffle(self._train_data)
        data = data_reader.read_file('test').data()[:4000]
        #data = chaos_data_reader.read_file('test').data()
        self._test_data = data

    def train_data(self):
        return self._train_data

    def test_data(self):
        return self._test_data
