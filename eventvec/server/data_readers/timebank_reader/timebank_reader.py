import os
from typing import List

from eventvec.server.data_readers.abstract import AbstractDataReader
from eventvec.server.data_readers.timebank_reader.timebank_model.timebank_document import TimebankDocument  # noqa
from eventvec.server.data_readers.timebank_reader.timebank_model.timebank_event import TimebankEvent  # noqa


class TimeMLDataReader(AbstractDataReader):
    def __init__(self):
        super().__init__()
        self.folder = self.config.timebank_data_location()

    def list_extra(self):
        """
            Lists the files in extra
        """
        relative_extra_folder_path = 'data/extra'
        absolute_extra_folder_path = os.path.join(self.folder,
                                                  relative_extra_folder_path)
        filelist = [os.path.join(absolute_extra_folder_path, i)
                    for i in os.listdir(absolute_extra_folder_path)]
        return filelist

    def read_file(self, filepath):
        with open(filepath) as f:
            return f.read()

    def xml2timebank_document(self, filecontents):
        timebank_document = TimebankDocument.from_xml(filecontents)
        return timebank_document

    def timebank_documents(self) -> List[TimebankDocument]:
        files = self.list_extra()
        timebank_documents = []
        for file in files:
            content = self.read_file(file)
            timebank_document = self.xml2timebank_document(content)
            timebank_documents.append(timebank_document)
        return timebank_documents


if __name__ == '__main__':
    tmdr = TimeMLDataReader()
    tmdr.timebank_documents()
