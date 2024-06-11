import csv
import json
from eventvec.server.config import Config
from jadelogs import JadeLogger

from eventvec.server.data.mnli.mnli_datamodels.mnli_datum import MNLIDatum
from eventvec.server.data.mnli.mnli_datamodels.mnli_data import MNLIData

label_dict = {
    'n': 'neutral',
    'c': 'contradiction',
    'e': 'entailment',
}

class ChaosMNLISyntaxDataReader:
    def __init__(self):
        config = Config.instance()
        self._mnli_file = 'chaos_mnli_syntax.jsonl'
        self._jade_logger = JadeLogger()

    def mnli_file_list(self):
        return [self._mnli_file]

    def read_file(self, filename):
        fullpath = self._jade_logger.file_manager.data_filepath(self._mnli_file)
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
                datum.set_uid(jsonl['uid'])
                data.add_datum(datum)
        return data
