from typing import Dict
import csv
import os
from jadelogs import JadeLogger

from eventvec.server.tasks.subordinate.datareader.datamodel import SubordinateRow
from eventvec.server.data.abstract import AbstractDatareader
from eventvec.server.config import Config


files = {
    'temporal_subordinate_said': 'temporal_subordinate_said.tsv',
    'temporal_subordinate_stated': 'temporal_subordinate_stated.tsv',
    'temporal_subordinate_suggested': 'temporal_subordinate_suggested.tsv',
    'temporal_subordinate_insinuated': 'temporal_subordinate_insinuated.tsv',
    'temporal_subordinate_retiring': 'temporal_subordinate_retiring.tsv',
    'temporal_subordinate_jogging_unembedded': 'temporal_subordinate_jogging_unembedded.tsv',
    'temporal_subordinate_dying': 'temporal_subordinate_dying.tsv',
}

class SubordinateTemporalDatareader(AbstractDatareader):
    def __init__(self):
        super().__init__()
        self._config = Config.instance()
        self.folder = self._config.subordinate_data_location()
        self._jade_logger = JadeLogger()

    def data(self, name):
        path = self._config.subordinate_data_location()
        full_path = os.path.join(path, files[name])
        data = []
        with open(full_path, 'r') as f:
            reader = csv.reader(f, delimiter='\t')
            for ri, r in enumerate(reader):
                if r[0] != '' and ri > 0:
                    data.append(SubordinateRow.from_csv_row(r))
        return data

if __name__ == '__main__':
    std = SubordinateTemporalDatareader()
    data = std.data()
