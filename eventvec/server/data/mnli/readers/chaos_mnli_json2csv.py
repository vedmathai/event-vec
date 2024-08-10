import os
from typing import Dict
import json
from collections import defaultdict
import numpy as np
import csv
import numpy


from eventvec.server.data.mnli.chaos_mnli_data.chaos_mnli_datum import ChaosMNLIDatum
from eventvec.server.data.mnli.chaos_mnli_data.chaos_mnli_data import ChaosMNLIData
from eventvec.server.data.abstract import AbstractDatareader


class ChaosMNLIDatareader(AbstractDatareader):
    def __init__(self):
        super().__init__()
        self.folder = self._config.chaos_mnli_data_location()

    def read_file(self, filepath):
        with open(filepath) as f:
            return f.read()

    def read_data(self) -> Dict:
        filepath = self.folder
        data = []
        with open(filepath) as f:
            for line in f:
                json_data = json.loads(line)
                chaos_mnli_data = ChaosMNLIDatum.from_json(json_data)
                data.append(chaos_mnli_data)
        return data

    def convert(self):
        data = self.read_data()
        with open('/home/lalady6977/Downloads/chaos_mnli.csv', 'wt', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(['body_parent', 'body_child', 'agreement'])
            for datum in data:
                    csvwriter.writerow([datum.premise(), datum.hypothesis(), datum.majority_label()])


if __name__ == '__main__':
    fr = ChaosMNLIDatareader()
    data = fr.convert()
