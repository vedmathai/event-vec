
class TSQAAnnotationAnswer:
    def __init__(self):
        self._para = None
        self._from_token = None
        self._end_token = None
        self._answer = None

    def para(self):
        return self._para

    def from_token(self):
        return self._from_token

    def end_token(self):
        return self._end_token

    def answer(self):
        return self._answer

    def set_para(self, para):
        self._para = para

    def set_from_token(self, from_token):
        self._from_token = from_token

    def set_end_token(self, end_token):
        self._end_token = end_token

    def set_answer(self, answer):
        self._answer = answer

    def to_dict(self):
        return {
            "para": self.para(),
            "from_token": self.from_token(),
            "end_token": self.end_token(),
            "answer": self.answer(),
        }

    @staticmethod
    def from_dict(val):
        tsqa_annotation_answer = TSQAAnnotationAnswer()
        tsqa_annotation_answer.set_para(val['para'])
        tsqa_annotation_answer.set_from_token(val['from_token'])
        tsqa_annotation_answer.set_end_token(val['end_token'])
        tsqa_annotation_answer.set_answer(val['answer'])
        return tsqa_annotation_answer

    @staticmethod
    def from_dataset(val):
        tsqa_annotation_answer = TSQAAnnotationAnswer()
        tsqa_annotation_answer.set_para(val['para'])
        tsqa_annotation_answer.set_from_token(val['from'])
        tsqa_annotation_answer.set_end_token(val['end'])
        tsqa_annotation_answer.set_answer(val['answer'])
        return tsqa_annotation_answer