from eventvec.server.data_readers.time_sensitive_qa_reader.time_sensitive_qa_datamodel.time_sensitive_qa_annotation_answer import TSQAAnnotationAnswer


class TSQAAnnotationQuestion:
    def __init__(self):
        self._time_1 = None
        self._time_2 = None
        self._tsqa_annotation_answers = []
        self._idx = None

    def idx(self):
        return self._idx

    def time_1(self):
        return self._time_1

    def time_2(self):
        return self._time_2

    def tsqa_annotation_answers(self):
        return self._tsqa_annotation_answers

    def set_idx(self, idx):
        self._idx = idx

    def set_time_1(self, time_1):
        self._time_1 = time_1

    def set_time_2(self, time_2):
        self._time_2 = time_2

    def set_tsqa_annotation_answers(self, tsqa_annotation_answers):
        self._tsqa_annotation_answers = tsqa_annotation_answers

    def to_dict(self):
        return {
            "idx": self.idx(),
            "time_1": self.time_1(),
            "time_2": self.time_2(),
            "tsqa_annotation_answer": [
                i.to_dict() for i in self.tsqa_annotation_answers()
            ]
        }

    @staticmethod
    def from_dict(val):
        tsqa_annotation_question = TSQAAnnotationQuestion()
        tsqa_annotation_answers.set_idx(val['idx'])
        tsqa_annotation_question.set_time_1(val['time_1'])
        tsqa_annotation_question.set_time_2(val['time_2'])
        tsqa_annotation_answers = [
            TSQAAnnotationAnswer.from_dict(i) for i in val['tsqa_annotation_answers']
        ]
        tsqa_annotation_question.set_tsqa_annotation_answers(tsqa_annotation_answers)
        return tsqa_annotation_question

    @staticmethod
    def from_dataset(idx, val):
        tsqa_annotation_question = TSQAAnnotationQuestion()
        tsqa_annotation_question.set_idx(idx)
        tsqa_annotation_question.set_time_1(val[0][0])
        tsqa_annotation_question.set_time_2(val[0][1])
        annotation_answers = [TSQAAnnotationAnswer.from_dataset(i) for i in val[1]]
        tsqa_annotation_question.set_tsqa_annotation_answers(annotation_answers)
        return tsqa_annotation_question
