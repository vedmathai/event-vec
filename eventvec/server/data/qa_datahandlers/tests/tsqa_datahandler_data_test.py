import unittest

from eventvec.server.config import Config
from eventvec.server.data_handlers.qa_datahandlers.tsqa_datahandler.tsqa_datahandler import TSQADatahandler
from eventvec.server.model.qa_models.datamodel.qa_dataset import QADataset


class TSQA2QA_test(unittest.TestCase):
    def setUp(self) -> None:
        self.config = Config.instance()
        path = "eventvec/server/data_readers/time_sensitive_qa_reader/time_sensitive_qa_datamodel/test/time_sensitive_qa_data"
        self.config.set_tsqa_data_location(path)
        filename = "test_sensitive_data.json"
        self.config.set_tsqa_file_names([filename])
        self.config.set_tsqa_file2annotation_map({
            "test_sensitive_data.json": "annotation_test_data.json"
        })


    def test_tsqa2qa_converter(self):
        tsqa_datahandler = TSQADatahandler()
        qa_data = tsqa_datahandler.qa_data()
        qa_data_new = QADataset.from_dict(qa_data.to_dict())
        self.assertEqual(qa_data.to_dict(), qa_data_new.to_dict())


if __name__ == '__main__':
    unittest.main()