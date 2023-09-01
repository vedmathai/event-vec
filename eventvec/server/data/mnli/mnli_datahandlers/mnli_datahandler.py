import csv
import json
from eventvec.server.config import Config


class MNLIDatahandler:
    def __init__(self):
        config = Config.instance()
        self._mnli_file = config.mnli_data_location()

    def mnli_file_list(self):
        return [self._mnli_file]

    def read_file(self, filename):
        fullpath = self._mnli_file
        elements = []
        with open(fullpath) as f:
            for line in f:
                jsonl = json.loads(line)
                elements.append(jsonl['sentence1'])
                elements.append(jsonl['sentence2'])
        return elements
