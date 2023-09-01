import os
import json

from eventvec.server.config import Config


class WikiDatahandler:
    def __init__(self):
        config = Config.instance()
        self._wiki_folder = config.wiki_data_location()

    def wiki_file_list(self):
        return os.listdir(self._wiki_folder)

    def read_file(self, filename):
        fullpath = os.path.join(self._wiki_folder, filename)
        lines = []
        with open(fullpath) as f:
            for line in f:
                text = json.loads(line)['text']
                lines.append(text)
        return lines
