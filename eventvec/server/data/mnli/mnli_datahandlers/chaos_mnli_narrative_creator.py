import csv
import json
from eventvec.server.config import Config
from jadelogs import JadeLogger

from narrativity.graph2sentence.graph2sentence import sentence2simple
from eventvec.server.data.mnli.mnli_datamodels.mnli_datum import MNLIDatum
from eventvec.server.data.mnli.mnli_datamodels.mnli_data import MNLIData

label_dict = {
    'n': 'neutral',
    'c': 'contradiction',
    'e': 'entailment',
}

class ChaosMNLIDatareader:
    def __init__(self):
        config = Config.instance()
        self._mnli_file = config.chaos_mnli_data_location()
        self._jade_logger = JadeLogger()

    def mnli_file_list(self):
        return [self._mnli_file]

    def read_file(self, filename):
        fullpath = self._mnli_file
        data = []
        updated_syntax_file = self._jade_logger.file_manager.data_filepath('chaos_mnli_syntax.jsonl')
        with open(updated_syntax_file, 'wt') as g:
            with open(fullpath) as f:
                for linei, line in enumerate(f):
                    print(linei)
                    jsonl = json.loads(line)
                    premise = jsonl['example']['premise']
                    hypothesis = jsonl['example']['hypothesis']
                    jsonl['example']['premise'] = sentence2simple(premise)
                    jsonl['example']['hypothesis'] = sentence2simple(hypothesis)
                    g.write(json.dumps(jsonl) + '\n')

if __name__ == '__main__':
    ChaosMNLIDatareader().read_file('test')