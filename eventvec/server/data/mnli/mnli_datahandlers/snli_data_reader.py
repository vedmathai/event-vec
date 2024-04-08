import csv
import json
import os

from eventvec.server.config import Config

from eventvec.server.data.mnli.mnli_datamodels.mnli_datum import MNLIDatum
from eventvec.server.data.mnli.mnli_datamodels.mnli_data import MNLIData

filenames = {
    'train': 'snli_1.0_train.jsonl',
    'test': 'snli_1.0_test.jsonl',
}

class SNLIDataReader:
    def __init__(self):
        config = Config.instance()
        self._snli_folder = config.snli_data_location()

    def snli_file_list(self):
        return ['train']

    def read_file(self, train_test='train'):
        filename = filenames[train_test]
        fullpath = os.path.join(self._snli_folder, filename)
        data = MNLIData()
        with open(fullpath) as f:
            for line in f:
                datum = MNLIDatum()
                jsonl = json.loads(line)
                datum.set_label(jsonl['gold_label'])
                datum.set_sentence_1(jsonl['sentence1'])
                datum.set_sentence_2(jsonl['sentence2'])
                datum.set_annotator_labels(jsonl['annotator_labels'])
                if datum.label() != '-' and len(datum.sentence_1().split()) < 20:
                    data.add_datum(datum)
        return data
