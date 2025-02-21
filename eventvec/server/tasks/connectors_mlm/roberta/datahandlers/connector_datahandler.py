
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
        seen = set()
        dedupe_data = []
        random.shuffle(data)
        for d in data:
            prefix = ' '.join(d.para().split()[:3])
            if prefix not in seen:
                dedupe_data.append(d)
                seen.add(prefix)
        self._train_data = data[:int(len(dedupe_data) * 0.8)]
        self._test_data = data[int(len(dedupe_data) * 0.8):]
        return dedupe_data

    def train_data(self):
        return self._train_data

    def test_data(self):
        return self._test_data
