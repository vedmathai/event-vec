import os
from typing import List

from eventvec.server.data.abstract import AbstractDatareader
from eventvec.server.data.timebank.timebank_reader.timebank_model.timebank_document import TimebankDocument  # noqa


class TE3GoldDatareader(AbstractDatareader):
    def __init__(self):
        super().__init__()
        self._silver_folder = self._config.te3_gold_data_location()

    def list_folder(self, run_name):
        filelist = [os.path.join(self._silver_folder, i)
                    for i in os.listdir(self._silver_folder)]
        return filelist

    def read_file(self, filepath):
        with open(filepath) as f:
            return f.read()

    def xml2timebank_document(self, filecontents):
        timebank_document = TimebankDocument.from_xml(filecontents)
        return timebank_document

    def timebank_documents(self, run_name) -> List[TimebankDocument]:
        files = self.list_folder(run_name)
        timebank_documents = []
        for file in files:
            content = self.read_file(file)
            timebank_document = self.xml2timebank_document(content)
            timebank_documents.append(timebank_document)
        return timebank_documents
