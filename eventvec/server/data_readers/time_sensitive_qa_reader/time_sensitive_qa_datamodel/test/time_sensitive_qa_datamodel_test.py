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
        self.tsqa_dr = TSQADataReader()


    def test_sqa_data_reader(self):
        tsqa_docs = self.tsqa_dr.tsqa_documents()
        tsqa_doc = tsqa_docs[0]
        self.assertEqual(len(tsqa_doc.tsqa_data()), 2)
        datum = tsqa_doc.tsqa_data()[0]
        self.assertGreater(len(datum.paragraphs()), 0)


if __name__ == '__main__':
    unittest.main()