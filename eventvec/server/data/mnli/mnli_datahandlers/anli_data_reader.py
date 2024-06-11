import csv
import json
import os

from eventvec.server.config import Config

from eventvec.server.data.mnli.mnli_datamodels.mnli_datum import MNLIDatum
from eventvec.server.data.mnli.mnli_datamodels.mnli_data import MNLIData

filenames = {
    'train': 'R3/train.jsonl',
    'test': 'R3/test.jsonl',
}

labels2label = {
    'e': 'entailment',
    'n': 'neutral',
    'c': 'contradiction',
}

class ANLIDataReader:
    def __init__(self):
        config = Config.instance()
        self._anli_folder = config.anli_data_location()

    def anli_file_list(self):
        return ['train']

    def read_file(self, train_test='train'):
        filename = filenames[train_test]
        fullpath = os.path.join(self._anli_folder, filename)
        data = MNLIData()
        with open(fullpath) as f:
            for line in f:
                datum = MNLIDatum()
                jsonl = json.loads(line)
                datum.set_label(labels2label[jsonl['label']])
                datum.set_sentence_1(jsonl['context'])
                datum.set_sentence_2(jsonl['hypothesis'])
                datum.set_uid(jsonl['uid'])
                data.add_datum(datum)
                
        return data
