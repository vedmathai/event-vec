import csv
import json
import os
from jadelogs import JadeLogger

from eventvec.server.config import Config
from eventvec.server.data.mnli.mnli_datamodels.mnli_datum import MNLIDatum
from eventvec.server.data.mnli.mnli_datamodels.mnli_data import MNLIData
from eventvec.server.featurizers.contrast_featurizer.contrast_featurizer import ContrastFeaturizer

filenames = {
    'train': 'multinli_1.0_train.jsonl',
    'test': 'multinli_1.0_dev_matched.jsonl',
}

class MNLIDataReader:
    def __init__(self):
        config = Config.instance()
        self._mnli_folder = config.mnli_data_location()
        self._jade_logger = JadeLogger()


    def mnli_file_list(self):
        return ['train']

    def read_file(self, train_test='train'):
        filename = filenames[train_test]
        fullpath = os.path.join(self._mnli_folder, filename)
        updated_syntax_file = self._jade_logger.file_manager.data_filepath('mnli_syntax.jsonl')
        fc = ContrastFeaturizer()
        count = 0
        with open(updated_syntax_file, 'wt') as g:
            with open(fullpath) as f:
                for linei, line in enumerate(f):
                    if linei == 11000:
                        break
                    if linei % 500 == 0:
                        print(linei)
                    jsonl = json.loads(line)
                    premise = jsonl['sentence1']
                    hypothesis = jsonl['sentence2']
                    jsonl['sentence1'] = fc.featurize(premise)
                    jsonl['sentence2'] = fc.featurize(hypothesis)
                    if 'CONTRASTS' in jsonl['sentence1'] + jsonl['sentence2']:
                        count += 1
                    g.write(json.dumps(jsonl) + '\n')

        print(count)
if __name__ == '__main__':
    MNLIDataReader().read_file('train')