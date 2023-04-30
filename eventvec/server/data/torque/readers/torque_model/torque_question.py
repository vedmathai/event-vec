from eventvec.server.data.torque.readers.torque_model.torque_events_answer import TorqueEventsAnswer

class TorqueQuestion:
    def __init__(self):
        self._question = None
        self._answer = None
        self._alternate_answers = []

    def answer(self):
        return self._answer

    def question(self):
        return self._question
    
    def alternate_answers(self):
        return self._alternate_answers

    def set_answer(self, answer):
        self._answer = answer

    def set_question(self, question):
        self._question = question

    def set_alternate_answers(self, alternate_answers):
        self._alternate_answers = alternate_answers

    @staticmethod
    def from_dict(val):
        question = TorqueQuestion()
        question.set_answer(TorqueEventsAnswer.from_dict(val['answer']))
        question.set_question(val['question'])
        return question
    
    @staticmethod
    def from_eval_dict(k, v):
        question = TorqueQuestion()
        question.set_answer(TorqueEventsAnswer.from_dict(v['answer']))
        alternate_answers = [TorqueEventsAnswer.from_dict(i) for i in v['individual_answers']]
        question.set_alternate_answers(alternate_answers)
        question.set_question(k)
        return question

    def to_dict(self):
        return {
            "answer": self.answer().to_dict(),
            "question": self.questions(),
            "alternate_answers": [i.to_dict() for i in self.alternate_answers()]
        }
