import json
import os
from typing import List

from eventvec.server.data.abstract import AbstractDatareader
from eventvec.server.data.torque.readers.torque_model.torque_dataset import TorqueDataset


class TorqueDataReader(AbstractDatareader):
    def __init__(self):
        super().__init__()
        self._folder = self._config.torque_abs_data_location()
        self._file_names = self._config.torque_data_file_names()

    def read_file(self, filepath):
        with open(filepath) as f:
            return f.read()

    def torque_train_dataset(self) -> TorqueDataset:
        datasets = []
        filename = self._file_names["train"]
        abs_filename = os.path.join(self._folder, filename)
        with open(abs_filename) as f:
            ds = TorqueDataset.from_train_dict(json.load(f))
            datasets.append(ds)
        return datasets
    
    def torque_test_eval_dataset(self, split_type) -> TorqueDataset:
        datasets = []
        filename = self._file_names[split_type]
        abs_filename = os.path.join(self._folder, filename)
        with open(abs_filename) as f:
            ds = TorqueDataset.from_eval_dict(json.load(f))
            datasets.append(ds)
        return datasets

    def torque_eval_dataset(self):
        return self.torque_test_eval_dataset('eval')