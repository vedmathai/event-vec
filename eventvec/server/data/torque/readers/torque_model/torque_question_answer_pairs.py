from eventvec.server.data.torque.readers.torque_model.torque_question import TorqueQuestion


class TorqueQuestionAnswerPairs:
    def __init__(self):
        self._questions = None

    def questions(self):
        return self._questions

    def set_questions(self, questions):
        self._questions = questions

    @staticmethod
    def from_dict(val):
        tqap = TorqueQuestionAnswerPairs()
        tqap.set_questions(TorqueQuestion.from_dict(i) for i in val)
        return tqap
    
    @staticmethod
    def from_eval_dict(val):
        tqap = TorqueQuestionAnswerPairs()
        tqap.set_questions(TorqueQuestion.from_eval_dict(k, v) for k, v in val)
        return tqap
    
    def to_dict(self):
        return {
            "answer": self.answer().to_dict(),
            "passage_id": self.passage_id(),
        }
