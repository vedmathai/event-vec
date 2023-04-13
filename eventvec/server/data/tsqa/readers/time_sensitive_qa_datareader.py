import json
import os
from typing import List

from eventvec.server.data_readers.abstract import AbstractDataReader
from eventvec.server.data_readers.time_sensitive_qa_reader.time_sensitive_qa_datamodel.time_sensitive_qa_document import TSQADocument  # noqa
from eventvec.server.data_readers.time_sensitive_qa_reader.time_sensitive_qa_datamodel.time_sensitive_qa_datum import TSQADatum  # noqa
from eventvec.server.data_readers.time_sensitive_qa_reader.time_sensitive_qa_datamodel.time_sensitive_qa_annotation_document import TSQAAnnotationDocument  # noqa


class TSQADataReader(AbstractDataReader):
    def __init__(self):
        super().__init__()
        self.folder = self.config.tsqa_data_location()
        self._file_names = self.config.tsqa_file_names()

    def list_extra(self):
        """
            Lists the files in extra
        """
        relative_extra_folder_path = 'dataset'
        absolute_extra_folder_path = os.path.join(self.folder,
                                                  relative_extra_folder_path)
        filelist = []
        for i in os.listdir(absolute_extra_folder_path):
            if i in self._file_names:
                filepath = os.path.join(absolute_extra_folder_path, i)
                annotation_filepath = os.path.join(absolute_extra_folder_path, self.config.tsqa_file2annotation_map()[i])
                filelist.append((filepath, annotation_filepath))
        return filelist

    def read_file(self, filepath):
        with open(filepath) as f:
            return f.read()

    def tsqa_read_annotation_document(self, filepath):
        tsqa_annotation_document = TSQAAnnotationDocument.from_dataset(filepath)
        return tsqa_annotation_document

    def tsqa_documents(self) -> List[TSQADocument]:
        files = self.list_extra()
        tqda_documents = []
        for file, annotation_filepath in files:
            annotation_document = self.tsqa_read_annotation_document(annotation_filepath) 
            content = self.read_file(file)
            document = TSQADocument()
            for line in content.strip().split('\n'):
                datum_dict = json.loads(line)
                datum = TSQADatum.from_dict(datum_dict)
                document.add_tsqa_datum(datum)
            document.set_tsqa_annotation_document(annotation_document)
            tqda_documents.append(document)

        return tqda_documents


if __name__ == '__main__':
    tsqa = TSQADocument()
    tsqa = tsqa.tsqa_documents()
