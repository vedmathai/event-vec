import csv
import json
import os
import uuid

from eventvec.server.config import Config

from eventvec.server.data.mnli.mnli_datamodels.mnli_datum import MNLIDatum
from eventvec.server.data.mnli.mnli_datamodels.mnli_data import MNLIData


class ConnectorNLIDatareader:
    def __init__(self):
        config = Config.instance()
        self._cnli_filename = config.connector_nli_data_location()

    def read_file(self, train_test='train'):
        filename = self._cnli_filename
        data = MNLIData()
        with open(filename) as f:
            reader = csv.reader(f, delimiter='\t')
            for line_i, line in enumerate(reader):
                datum = MNLIDatum()
                datum.set_label(line[6])
                if len(line[4]) > 0:
                    datum.set_sentence_1(line[4])
                    datum.set_sentence_2(line[5])
                    datum.set_uid(line[0])
                    datum.set_type('{}_{}_{}'.format(line[1], line[2], line[3]))
                    data.add_datum(datum)
        return data
