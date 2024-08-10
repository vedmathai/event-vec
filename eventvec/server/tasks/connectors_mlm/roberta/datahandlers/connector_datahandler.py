
import random
from eventvec.server.data.connectors.datareaders.connector_datareader import ConnectorsDatareader


class ConnectorsDatahandler():
    def __init__(self):
        self._data_readers = {
            'connectors': ConnectorsDatareader,
        } 

    def load(self, run_config):
        data_reader = self._data_readers['connectors']
        data = data_reader().data().data()
        random.shuffle(data)
        self._train_data = data[:int(len(data) * 0.8)]
        self._test_data = data[int(len(data) * 0.8):]
        return data

    def train_data(self):
        return self._train_data

    def test_data(self):
        return self._test_data
