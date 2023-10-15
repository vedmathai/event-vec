import os
from typing import List

from eventvec.server.data.abstract import AbstractDatareader
from eventvec.server.data.timebank.timebank_reader.timebank_model.timebank_document import TimebankDocument  # noqa


class AquaintDatareader(AbstractDatareader):
    def __init__(self):
        super().__init__()
        self._gold_folder = self._config.aquaint_data_location()

    def list_folder(self, run_name=None):
        filelist = [os.path.join(self._gold_folder, i)
                    for i in os.listdir(self._gold_folder)]
        return filelist

    def read_file(self, filepath):
        with open(filepath) as f:
            return f.read()

    def xml2timebank_document(self, filecontents):
        timebank_document = TimebankDocument.from_xml(filecontents)
        return timebank_document

    def timebank_documents(self, run_name=None) -> List[TimebankDocument]:
        files = self.list_folder(run_name)
        timebank_documents = []
        for file in files:
            content = self.read_file(file)
            timebank_document = self.xml2timebank_document(content)
            timebank_documents.append(timebank_document)
        return timebank_documents

    def timebank_documents_contents(self, filename):
        files = self.list_folder('train')
        contents = []
        for file in files:
            content = self.read_file(file)
            timebank_document = self.xml2timebank_document(content)
            contents.append(timebank_document.text())
        return contents
