from eventvec.server.data.tsqa.readers.time_sensitive_qa_datareader import TSQADataReader
from eventvec.server.data.tsqa.datahandlers.tsqa_converter import TSQAConverter


class TSQADatahandler:
    def __init__(self):
        self._tsqa_converter = TSQAConverter()
        self._tsqa_datareader = TSQADataReader()

    def qa_data(self):
        tsqa_documents = self._tsqa_datareader.tsqa_documents()
        qa_data = self._tsqa_converter.convert(tsqa_documents)
        return qa_data
