import os
from typing import Dict
import json
from collections import defaultdict
import numpy as np
import csv

from eventvec.server.data.factuality.factuality_datamodels.belief_datum import BeliefDatum
from eventvec.server.data.factuality.factuality_datamodels.belief_data import BeliefData
from eventvec.server.data.abstract import AbstractDatareader


class FactualityReader(AbstractDatareader):
    def __init__(self):
        super().__init__()
        self.folder = self._config.factuality_data_location()

    def belief_files(self):
        filepaths = ["tempeval.belief.1.json", "tempeval.belief.2.json"]
        for filepath in filepaths:
            yield os.path.join(self.folder, "data/TBAQ/crowdflower", filepath)
    
    def read_file(self, filepath):
        with open(filepath) as f:
            return f.read()

    def belief_data(self) -> Dict:
        filepaths = self.belief_files()
        data = []
        data = BeliefData()
        for filepath in filepaths:  
            print('start', len(data.data()))
            with open(filepath) as f:
                for line in f:
                    json_data = json.loads(line)
                    belief_datum = BeliefDatum.parse_raw_data(json_data)
                    data.add_datum(belief_datum)
            print('finish', len(data.data()))
        return data
