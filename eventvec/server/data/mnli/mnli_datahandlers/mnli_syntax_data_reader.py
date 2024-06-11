import csv
import json
import os
from jadelogs import JadeLogger

from eventvec.server.config import Config

from eventvec.server.data.mnli.mnli_datamodels.mnli_datum import MNLIDatum
from eventvec.server.data.mnli.mnli_datamodels.mnli_data import MNLIData

filenames = {
    'train': 'mnli_syntax.jsonl',
}

class MNLISyntaxDataReader:
    def __init__(self):
        config = Config.instance()
        self._jade_logger = JadeLogger()

    def mnli_file_list(self):
        return ['train']

    def read_file(self, train_test='train'):
        filename = filenames[train_test]
        fullpath = self._jade_logger.file_manager.data_filepath(filename)
        data = MNLIData()
        with open(fullpath) as f:
            for line in f:
                datum = MNLIDatum()
                jsonl = json.loads(line)
                datum.set_label(jsonl['gold_label'])
                datum.set_sentence_1(jsonl['sentence1'])
                datum.set_sentence_2(jsonl['sentence2'])
                datum.set_annotator_labels(jsonl['annotator_labels'])
                #if datum.label() != '-' and len(datum.sentence_1().split()) < 20:
                data.add_datum(datum)
        return data
