import os
import json

from eventvec.server.config import Config


class PolitosphereDatahandler:
    def __init__(self):
        config = Config.instance()
        self._politosphere_folder = config.politosphere_data_location()

    def politosphere_file_list(self):
        files = os.listdir(self._politosphere_folder)
        filelist = []
        for file in files:
            full_path = os.path.join(self._politosphere_folder, file)
            filelist.append(full_path)
        return filelist

    def read_file(self, filename):
        filelist = self.politosphere_file_list()
        elements = []
        for fullpath in filelist:
            with open(fullpath) as f:
                for line in f:
                    json_line = json.loads(line)
                    elements.append(json_line['body'])
        return elements
