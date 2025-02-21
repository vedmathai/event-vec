from typing import Dict
import csv
import os
from jadelogs import JadeLogger

from eventvec.server.tasks.subordinate.datareader.datamodel import SubordinateRow
from eventvec.server.data.abstract import AbstractDatareader
from eventvec.server.config import Config


files = {
    'temporal_subordinate_said': 'temporal_subordinate_said.csv',
    'temporal_subordinate_stated': 'temporal_subordinate_stated.csv',
    'temporal_subordinate_suggested': 'temporal_subordinate_suggested.csv',
    'temporal_subordinate_insinuated': 'temporal_subordinate_insinuated.csv',
}

class SubordinateTemporalDatareader(AbstractDatareader):
    def __init__(self):
        super().__init__()
        self._config = Config.instance()
        self.folder = self._config.subordinate_data_location()
        self._jade_logger = JadeLogger()

    def data(self):
        path = self._config.subordinate_data_location()
        data = []
        with open(path, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for ri, r in enumerate(reader):
                if r[0] != '':
                    data.append(SubordinateRow.from_csv_row(r))
        return data

if __name__ == '__main__':
    std = SubordinateTemporalDatareader()
    data = std.data()
