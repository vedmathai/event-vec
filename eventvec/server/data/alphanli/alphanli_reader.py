import csv
import json
import os

from eventvec.server.config import Config

from eventvec.server.data.alphanli.datamodel.alphanli_datum import AlphaNLIDatum
from eventvec.server.data.alphanli.datamodel.alpha_nli_data import AlphaNLIData

filenames = {
    'train': 'train.jsonl',
    'dev': 'dev.jsonl',
}

label_filenames = {
    'train': 'train-labels.lst',
    'dev': 'dev-labels.lst',

}

class AlphaNLIDataReader:
    def __init__(self):
        config = Config.instance()
        self._alphanli_folder = config.alphanli_data_location()

    def mnli_file_list(self):
        return ['train']

    def read_file(self, train_test='train'):
        filename = filenames[train_test]
        label_filename = label_filenames[train_test]
        file_fullpath = os.path.join(self._alphanli_folder, filename)
        label_fullpath = os.path.join(self._alphanli_folder, label_filename)
        data = AlphaNLIData()
        label_i2label = {}
        with open(label_fullpath) as f:
            reader = csv.reader(f, delimiter='\t')
            for rowi, row in enumerate(reader):
                label_i2label[rowi] = row[0]

        with open(file_fullpath) as f:
            for linei, line in enumerate(f):
                datum = AlphaNLIDatum()
                jsonl = json.loads(line)
                datum.set_label(label_i2label[linei])
                datum.set_uid(jsonl['story_id'])
                datum.set_obs_1(jsonl['obs1'])
                datum.set_obs_2(jsonl['obs2'])
                datum.set_hyp_1(jsonl['hyp1'])
                datum.set_hyp_2(jsonl['hyp2'])
                data.add_datum(datum)
        return data


if __name__ == '__main__':
    reader = AlphaNLIDataReader()
    data = reader.read_file('train')
    count = 0
    for datum in data.data():
        if datum.label() == '1':
            hyp = datum.hyp_1()
        elif datum.label() == '2':
            hyp = datum.hyp_2()
        if 'so' in hyp.split():
            count += 1