import numpy as np
import pprint
import re
from collections import defaultdict
import pprint


from eventvec.server.data.factuality.datahandlers.factuality_datahandler import FactualityDatahandler  # noqa


class FactualityRoBERTaDataHandler():
    def __init__(self):
        self._data_handler = FactualityDatahandler()

    def load(self):
        self._data_handler.load()

    def train_data(self):
        train_data = self._data_handler.train_data()
        return train_data

    def test_data(self):
        test_data = self._data_handler.test_data()
        return test_data
