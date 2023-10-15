import csv
import os

from eventvec.server.config import Config


class MAECDatahandler:
    def __init__(self):
        config = Config.instance()
        self._maec_folder = config.maec_data_location()

    def maec_file_list(self):
        subfolders = os.listdir(self._maec_folder)
        filelist = []
        for subfolder in subfolders:
            full_path = os.path.join(self._maec_folder, subfolder, 'text.txt')
            filelist.append(full_path)
        return filelist

    def read_file(self, filename):
        filelist = self.maec_file_list()
        elements = []
        for fullpath in filelist:
            with open(fullpath) as f:
                elements.append(f.read())
        return elements
