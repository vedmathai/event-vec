from eventvec.server.data_readers.time_sensitive_qa_reader.time_sensitive_qa_datareader import TSQADataReader
from eventvec.server.data_handlers.qa_datahandlers.tsqa_datahandler.tsqa_converter import TSQAConverter


class TSQADatahandler:
    def __init__(self):
        self._tsqa_converter = TSQAConverter()
        self._tsqa_datareader = TSQADataReader()

    def qa_data(self):
        tsqa_documents = self._tsqa_datareader.tsqa_documents()
        qa_data = self._tsqa_converter.convert(tsqa_documents)
        return qa_data
