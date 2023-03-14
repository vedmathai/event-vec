
from eventvec.server.data_readers.time_sensitive_qa_reader.time_sensitive_qa_datamodel.time_sensitive_qa_paragraph import TSQAParagraph


class TSQADatum:
    def __init__(self):
        self._idx = None
        self._question = None
        self._context = None
        self._targets = []
        self._paragraphs = []

    def idx(self):
        return self._idx

    def question(self):
        return self._question

    def context(self):
        return self._context

    def targets(self):
        return self._targets

    def paragraphs(self):
        return self._paragraphs

    def set_idx(self, idx):
        self._idx = idx

    def set_question(self, question):
        self._question = question

    def set_context(self, context):
        self._context = context

    def set_targets(self, targets):
        self._targets = targets

    def set_paragraphs(self, paragraphs):
        self._paragraphs = paragraphs

    @staticmethod
    def from_dict(val):
        tsqa_datum = TSQADatum()
        tsqa_datum.set_idx(val['idx'])
        tsqa_datum.set_question(val['question'])
        tsqa_datum.set_context(val['context'])
        tsqa_datum.set_targets(val['targets'])
        paragraphs = [TSQAParagraph.from_dict(i) for i in val['paragraphs']]
        tsqa_datum.set_paragraphs(paragraphs)
        return tsqa_datum

    def to_dict(self):
        return {
            'idx': self.idx(),
            'question': self.question(),
            'context': self.context(),
        }
