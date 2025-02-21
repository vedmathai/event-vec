
import random

from eventvec.server.tasks.event_ordering_nli.datareader.temporal_datareader import TemporalDatareader

class TemporalDatahandler():
    def __init__(self):
        self._data_readers = {
            'temporal': TemporalDatareader(),
        } 


    def load(self, run_config):
        data_reader = self._data_readers['temporal']
        data = data_reader.data(run_config.dataset())
        random.shuffle(data)
        self._train_data = data
        data = data_reader.data(run_config.test_dataset())
        random.shuffle(data)
        self._test_data = data


    def train_data(self):
        return self._train_data

    def test_data(self):
        return self._test_data
