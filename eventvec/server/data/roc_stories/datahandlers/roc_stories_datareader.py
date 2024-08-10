import os
import csv

from eventvec.server.config import Config

from eventvec.server.data.roc_stories.datahandlers.roc_datamodels.roc_datum import ROCDatum
from eventvec.server.data.roc_stories.datahandlers.roc_datamodels.roc_data import ROCData


class ROCStoriesDatareader:
    def __init__(self):
        config = Config.instance()
        self._roc_stories_folder = config.roc_stories_data_location()

    def read_file(self):
        fullpath = os.path.join(self._roc_stories_folder)
        lines = []
        with open(fullpath) as f:
            reader = csv.reader(f)
            for row in reader:
                datum = ROCDatum()
                datum.set_uid(row[0])
                datum.set_sentence_1(row[2])
                datum.set_sentence_2(row[3])
                datum.set_sentence_3(row[4])
                datum.set_sentence_4(row[5])
                datum.set_sentence_5(row[6])
                datum.set_title(row[0])
                lines.append(datum)
        return lines


if __name__ == '__main__':
    handler = ROCStoriesDatareader()
    data = handler.read_file()
    for datum in data:
        print(datum.to_dict())