import unittest

from eventvec.server.config import Config
from eventvec.server.data_readers.time_sensitive_qa_reader.time_sensitive_qa_datareader import TSQADataReader


class TestTimeSensitiveQADatamodel(unittest.TestCase):
    def setUp(self) -> None:
        self.config = Config.instance()
        path = "eventvec/server/data_readers/time_sensitive_qa_reader/time_sensitive_qa_datamodel/test/time_sensitive_qa_data"
        self.config.set_tsqa_data_location(path)
        filename = "test_sensitive_data.json"
        self.config.set_tsqa_file_names([filename])
        self.config.set_tsqa_file2annotation_map({
            "test_sensitive_data.json": "annotation_test_data.json"
        })
        self.tsqa_dr = TSQADataReader()


    def test_sqa_data_reader(self):
        tsqa_docs = self.tsqa_dr.tsqa_documents()
        tsqa_doc = tsqa_docs[0]
        self.assertEqual(len(tsqa_doc.tsqa_data()), 3)
        datum = list(tsqa_doc.tsqa_data().values())[0]
        self.assertGreater(len(datum.paragraphs()), 0)
        tsqa_annotation_doc = tsqa_doc.tsqa_annotation_document()
        for annotation in tsqa_annotation_doc.tsqa_annotations().values():
            for question in annotation.questions():
                datum = tsqa_doc.id2datum(question.idx())
                print(question.to_dict(), datum.question())


if __name__ == '__main__':
    unittest.main()