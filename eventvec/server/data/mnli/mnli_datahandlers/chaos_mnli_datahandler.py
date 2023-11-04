import csv
import json
from eventvec.server.config import Config


class ChaosMNLIDatahandler:
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
                element = []
                element.append(jsonl['gold_label'])
                element.append(jsonl['sentence1'])
                element.append(jsonl['sentence2'])
                elements.append(element)
        return elements
