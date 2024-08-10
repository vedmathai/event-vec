import csv
import json
from eventvec.server.config import Config
from jadelogs import JadeLogger
import re

from eventvec.server.data.mnli.mnli_datamodels.mnli_datum import MNLIDatum
from eventvec.server.data.mnli.mnli_datamodels.mnli_data import MNLIData
from eventvec.server.featurizers.contrast_featurizer.contrast_featurizer import ContrastFeaturizer


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
        fc = ContrastFeaturizer()
        count = 0
        data = []
        prog=re.compile(r'[0-9*]')
        with open(updated_syntax_file, 'wt') as g:
            with open(fullpath) as f:
                for linei, line in enumerate(f):
                    jsonl = json.loads(line)
                    premise = jsonl['example']['premise']
                    hypothesis = jsonl['example']['hypothesis']
                    jsonl['example']['premise'] = fc.featurize(premise)
                    jsonl['example']['hypothesis'] = fc.featurize(hypothesis)
                    if any(i in premise for i in ['tiny', 'big', 'huge']) or any(i in premise for i in ['tiny', 'big', 'huge']):
                        print(premise)
                        print(hypothesis)
                        print(jsonl['majority_label'])
                    if 'CONTRASTS' in jsonl['example']['premise'] + jsonl['example']['hypothesis']: 
                        data.append([jsonl['example']['premise'], jsonl['example']['hypothesis'], jsonl['majority_label']])
                        count += 1
                    g.write(json.dumps(jsonl) + '\n')
                    print(count, linei)
        with open('/home/lalady6977/Downloads/contrast.csv', 'wt') as f:
            writer = csv.writer(f, delimiter=',')
            for row in data:
                writer.writerow(row)
    
        
if __name__ == '__main__':
    ChaosMNLIDatareader().read_file('test')