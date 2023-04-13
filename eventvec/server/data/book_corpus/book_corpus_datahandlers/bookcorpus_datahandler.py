import os

from eventvec.server.config import Config


class BookCorpusDatahandler:
    def __init__(self):
        config = Config.instance()
        self._book_corpus_folder = config.book_corpus_data_location()

    def book_corpus_file_list(self):
        return os.listdir(self._book_corpus_folder)

    def read_file(self, filename):
        fullpath = os.path.join(self._book_corpus_folder, filename)
        with open(fullpath) as f:
            return f.read()
