import json
import os
from typing import List

from eventvec.server.data_readers.abstract import AbstractDataReader
from eventvec.server.data_readers.torque_reader.torque_model.torque_dataset import TorqueDataset


class TorqueDataReader(AbstractDataReader):
    def __init__(self):
        super().__init__()
        self._folder = self.config.torque_data_location()
        self._file_names = self.config.torque_data_file_names()

    def read_file(self, filepath):
        with open(filepath) as f:
            return f.read()

    def torque_dataset(self) -> TorqueDataset:
        datasets = []
        for filename in self._file_names:
            abs_filename = os.path.join(self._folder, filename)
            with open(abs_filename) as f:
                ds = TorqueDataset.from_dict(json.load(f))
                datasets.append(ds)
        return datasets


if __name__ == '__main__':
    tdr = TorqueDataReader()
    ds = tdr.torque_dataset()
    for d in ds[0].data():
        print(d.passage())
        
        print(d.events().answer().indices())