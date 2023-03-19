
class TSQAAnnotationAnswer:
    def __init__(self):
        self._para = None
        self._from_char = None
        self._end_char = None
        self._answer = None

    def para(self):
        return self._para

    def from_char(self):
        return self._from_char

    def end_char(self):
        return self._end_char

    def answer(self):
        return self._answer

    def set_para(self, para):
        self._para = para

    def set_from_char(self, from_char):
        self._from_char = from_char

    def set_end_char(self, end_char):
        self._end_char = end_char

    def set_answer(self, answer):
        self._answer = answer

    def to_dict(self):
        return {
            "para": self.para(),
            "from_char": self.from_char(),
            "end_char": self.end_char(),
            "answer": self.answer(),
        }

    @staticmethod
    def from_dict(val):
        tsqa_annotation_answer = TSQAAnnotationAnswer()
        tsqa_annotation_answer.set_para(val['para'])
        tsqa_annotation_answer.set_from_char(val['from_char'])
        tsqa_annotation_answer.set_end_char(val['end_char'])
        tsqa_annotation_answer.set_answer(val['answer'])
        return tsqa_annotation_answer

    @staticmethod
    def from_dataset(val):
        tsqa_annotation_answer = TSQAAnnotationAnswer()
        tsqa_annotation_answer.set_para(val['para'])
        tsqa_annotation_answer.set_from_char(val['from'])
        tsqa_annotation_answer.set_end_char(val['end'])
        tsqa_annotation_answer.set_answer(val['answer'])
        return tsqa_annotation_answer