
from eventvec.server.data_readers.time_sensitive_qa_reader.time_sensitive_qa_datamodel.time_sensitive_qa_paragraph import TSQAParagraph
from eventvec.server.data_readers.time_sensitive_qa_reader.time_sensitive_qa_datamodel.time_sensitive_qa_datum import TSQADatum


class TSQADocument:
    def __init__(self):
        self._tsqa_data_name = None
        self._tsqa_data = {}
        self._annotation_document = None
        self._tsqa_annotation_document = None

    def tsqa_data_name(self):
        return self._tsqa_data_name

    def tsqa_data(self):
        return self._tsqa_data

    def tsqa_annotation_document(self):
        return self._tsqa_annotation_document

    def set_tsqa_data_name(self, tsqa_data_name):
        self._tsqa_data_name = tsqa_data_name

    def set_tsqa_data(self, tsqa_data):
        self._tsqa_data = tsqa_data

    def set_tsqa_annotation_document(self, tsqa_annotation_document):
        self._tsqa_annotation_document = tsqa_annotation_document

    def add_tsqa_datum(self, tsqa_datum):
        self._tsqa_data[tsqa_datum.idx()] = tsqa_datum

    def id2datum(self, id):
        return self._tsqa_data[id]

    @staticmethod
    def from_dict(val):
        tsqa_data = TSQADocument()
        tsqa_data.set_tsqa_data_name(val['tsqa_data_name'])
        tsqa_datapoints = [TSQADatum.from_dict(i) for i in val['tsqa_data']]
        tsqa_datapoints = {i.idx(): i for i in tsqa_datapoints}
        tsqa_data.set_tsqa_data(tsqa_datapoints)
        return tsqa_data

    def to_dict(self):
        return {
            'tsqa_data_name': self.tsqa_data_name(),
            'tsqa_data': self.tsqa_data(),
        }
