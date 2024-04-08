import csv
import json
from eventvec.server.config import Config

from eventvec.server.data.mnli.mnli_datamodels.mnli_datum import MNLIDatum
from eventvec.server.data.mnli.mnli_datamodels.mnli_data import MNLIData

label_dict = {
    'n': 'neutral',
    'c': 'contradiction',
    'e': 'entailment',
}

class ChaosANLIDatareader:
    def __init__(self):
        config = Config.instance()
        self._anli_file = config.chaos_anli_data_location()

    def anli_file_list(self):
        return [self._anli_file]

    def read_file(self, filename):
        fullpath = self._anli_file
        data = MNLIData()
        with open(fullpath) as f:
            for line in f:
                jsonl = json.loads(line)
                datum = MNLIDatum()
                datum.set_label(label_dict[jsonl['majority_label']])
                datum.set_sentence_1(jsonl['example']['premise'])
                datum.set_sentence_2(jsonl['example']['hypothesis'])
                datum.set_entropy(jsonl['entropy'])
                datum.set_label_dist(jsonl['label_dist'])
                data.add_datum(datum)
        return data
