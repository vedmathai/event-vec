import csv
import json
import os

from eventvec.server.config import Config

from eventvec.server.data.mnli.mnli_datamodels.mnli_datum import MNLIDatum
from eventvec.server.data.mnli.mnli_datamodels.mnli_data import MNLIData

filenames = {
    'train': 'multinli_1.0_train.jsonl',
    'test': 'multinli_1.0_dev_matched.jsonl',
}

class MNLIDataReader:
    def __init__(self):
        config = Config.instance()
        self._mnli_folder = config.mnli_data_location()

    def mnli_file_list(self):
        return ['train']

    def read_file(self, train_test='train'):
        filename = filenames[train_test]
        fullpath = os.path.join(self._mnli_folder, filename)
        data = MNLIData()
        text_data = []
        with open(fullpath) as f:
            for line in f:
                datum = MNLIDatum()
                jsonl = json.loads(line)
                datum.set_label(jsonl['gold_label'])
                datum.set_sentence_1(jsonl['sentence1'])
                datum.set_sentence_2(jsonl['sentence2'])
                text_data.append(jsonl['sentence1'])
                text_data.append(jsonl['sentence2'])
                if datum.label() != '-':
                    data.add_datum(datum)
        return text_data
