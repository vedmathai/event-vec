import os
import json
from bs4 import BeautifulSoup


from eventvec.server.config import Config


class NYTDatahandler:
    def __init__(self):
        config = Config.instance()
        self._nyt_folder = config.nyt_data_location()

    def nyt_file_list(self):
        filepaths = []
        for i in os.listdir(self._nyt_folder):
            folderpath = os.path.join(self._nyt_folder, str(i))
            for j in os.listdir(folderpath):
                filepath = os.path.join(folderpath, j)
                filepaths.append(filepath)
        return filepaths

    def read_file(self, filename):
        fullpath = os.path.join(self._nyt_folder, filename)
        lines = []
        text = []
        with open(fullpath) as f:
            lines = f.read()
            soup = BeautifulSoup(lines, features="xml")
            ps = soup.find_all('p')
            for p in ps:
                text.append(p.text)
        text = ' '.join(text)
        return [text]
                
