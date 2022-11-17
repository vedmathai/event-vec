import os
from typing import List

from eventvec.server.data_readers.abstract import AbstractDataReader
from eventvec.server.data_readers.timebank_reader.timebank_model.timebank_document import TimebankDocument  # noqa


class TimeBankDenseDataReader(AbstractDataReader):
    def __init__(self):
        super().__init__()
        self.folder = self.config.timebank_dense_data_location()
        self._run2sub_folder = {
            "dev": "dev",
            "test": "test",
            "train": "train",
        }

    def list_folder(self, run_name):
        """
            Lists the files in extra
        """
        subfolder = self._run2sub_folder.get(run_name)
        absolute_extra_folder_path = os.path.join(self.folder,
                                                  subfolder)
        filelist = [os.path.join(absolute_extra_folder_path, i)
                    for i in os.listdir(absolute_extra_folder_path)]
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

    def train_documents(self) -> List[TimebankDocument]:
        return self.timebank_documents('train')

    def dev_documents(self) -> List[TimebankDocument]:
        return self.timebank_documents('dev')

    def test_documents(self) -> List[TimebankDocument]:
        return self.timebank_documents('test')
