import numpy as np
import pprint
import re
from collections import defaultdict
from transformers import BertTokenizer, RobertaTokenizer
import pprint
import random


from eventvec.server.data.timebank.datahandlers.timebank_data_handler import TimeBankBertDataHandler  # noqa
from eventvec.server.tasks.relationship_classification.featurizers.bert_featurizer import BERTLinguisticFeaturizer  # noqa

from eventvec.server.data.mnli.mnli_datahandlers.mnli_data_reader import MNLIDataReader  # noqa
from eventvec.server.data.mnli.mnli_datahandlers.chaos_mnli_data_reader import ChaosMNLIDatareader  # noqa


class NLIDataHandler():
    def __init__(self):
        self._data_reader = MNLIDataReader()
        self._chaos_data_reader = ChaosMNLIDatareader()

    def load(self):
        data = self._data_reader.read_file('train').data()
        random.seed(42)
        random.shuffle(data)
        train_size = 10000
        self._train_data = data[:train_size]
        data = self._chaos_data_reader.read_file('test').data()
        self._test_data = data

    def train_data(self):
        return self._train_data

    def test_data(self):
        return self._test_data
