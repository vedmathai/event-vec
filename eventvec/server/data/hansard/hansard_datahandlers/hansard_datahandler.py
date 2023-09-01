import csv

from eventvec.server.config import Config


class HansardDatahandler:
    def __init__(self):
        config = Config.instance()
        self._hansard_file = config.hansard_data_location()

    def hansard_file_list(self):
        return self._hansard_file

    def read_file(self, filename):
        fullpath = self._hansard_file
        elements = []
        with open(fullpath) as f:
            reader = csv.reader(f, delimiter=',')
            for r in reader:
                elements.append(r[1])
        return elements
