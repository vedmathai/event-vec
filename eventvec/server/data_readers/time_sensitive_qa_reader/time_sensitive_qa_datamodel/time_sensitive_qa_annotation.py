from eventvec.server.data_readers.time_sensitive_qa_reader.time_sensitive_qa_datamodel.time_sensitive_qa_annotation_question import TSQAAnnotationQuestion


class TSQAAnnotation:
    def __init__(self):
        self._index = []
        self._type = ""
        self._link = ""
        self._questions = []
        self._paragraphs = []

    def index(self):
        return self._index

    def type(self):
        return self._type

    def link(self):
        return self._link

    def questions(self):
        return self._questions

    def paragraphs(self):
        return self._paragraphs

    def set_index(self, index):
      self._index = index

    def set_type(self, type):
      self._type = type

    def set_link(self, link):
      self._link = link

    def set_questions(self, questions):
      self._questions = questions

    def add_question(self, question):
      self._questions.append(question)

    def set_paragraphs(self, paragraphs):
      self._paragraphs = paragraphs

    def tsqa_annotations(self):
        return self._tsqa_annotations

    def set_name(self, name):
        self._name = name

    def set_tsqa_annotations(self, tsqa_annotations):
        self._tsqa_annotations = tsqa_annotations

    @staticmethod
    def from_dict(val):
        tsqa_annotation = TSQAAnnotation()
        tsqa_annotation.set_index(val['index'])
        tsqa_annotation.set_type(val['type'])
        tsqa_annotation.set_link(val['link'])
        tsqa_annotation.set_questions(
            [TSQAAnnotationQuestion.from_val(i) for i in val['questions']]
        )
        tsqa_annotation.set_paragraphs(val['paragraphs'])
        return tsqa_annotation

    def to_dict(self):
        return {
            "index": self.index(),
            "type": self.type(),
            "link": self.link(),
            "questions": [i.to_dict() for i in self.questions()],
            "paragraphs": self.paragraphs(),
        }

    @staticmethod
    def from_dataset(val):
        tsqa_annotation = TSQAAnnotation()
        tsqa_annotation.set_index(val['index'])
        tsqa_annotation.set_type(val['type'])
        tsqa_annotation.set_link(val['link'])
        for ii, i in enumerate(val['questions']):
            idx = "{}#{}".format(tsqa_annotation.index(), ii)
            tsqa_annotation.add_question(TSQAAnnotationQuestion.from_dataset(idx, i))
        tsqa_annotation.set_paragraphs(val['paras'])
        return tsqa_annotation
