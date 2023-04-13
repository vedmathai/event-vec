import json
import os


from eventvec.server.data_readers.time_sensitive_qa_reader.time_sensitive_qa_datamodel.time_sensitive_qa_annotation import TSQAAnnotation  # noqa


class TSQAAnnotationDocument:
    def __init__(self):
        self._tsqa_annotations = []
        self._name = ""

    def name(self):
        return self._name

    def tsqa_annotations(self):
        return self._tsqa_annotations

    def set_name(self, name):
        self._name = name

    def set_tsqa_annotations(self, tsqa_annotations):
        self._tsqa_annotations = tsqa_annotations

    @staticmethod
    def from_dict(val):
        tsqa_annotation_document = TSQAAnnotationDocument()
        tsqa_annotation_document.set_name(val['name'])
        annotations = [TSQAAnnotation.from_val(i) for i in val['tsqa_annotations']]
        tsqa_annotation_document.set_tsqa_annotations({i.index(): i for i in annotations})
        return tsqa_annotation_document

    def to_dict(self):
        return {
            "name": self.name(),
            "tsqa_annotations": [i.to_dict() for i in self.tsqa_annotations()],
        }

    @staticmethod
    def from_dataset(filepath):
        filename = os.path.basename(filepath)
        with open(filepath) as f:
            val = json.load(f)
        tsqa_annotation_document = TSQAAnnotationDocument()
        tsqa_annotation_document.set_name(filename)
        annotations = [TSQAAnnotation.from_dataset(i) for i in val]
        tsqa_annotation_document.set_tsqa_annotations({i.index(): i for i in annotations})
        return tsqa_annotation_document
