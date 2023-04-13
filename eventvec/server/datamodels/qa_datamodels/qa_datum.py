from eventvec.server.model.qa_models.datamodel.qa_answer import QAAnswer


class QADatum:
    def __init__(self):
        self._id = None
        self._question = None
        self._context = []
        self._answers = []

    def id(self):
        return self._id

    def question(self):
        return self._question

    def context(self):
        return self._context

    def answers(self):
        return self._answers

    def set_id(self, id):
        self._id = id

    def set_question(self, question):
        self._question = question

    def set_context(self, context):
        self._context = context

    def set_answers(self, answers):
        self._answers = answers

    def add_answer(self, answer):
        self._answers.append(answer)

    def to_dict(self):
        return {
            "id": self.id(),
            "question": self.question(),
            "context": self.context(),
            "answers": [i.to_dict() for i in self.answers()],
        }

    @staticmethod
    def from_dict(val):
        qa_datum = QADatum()
        qa_datum.set_id(val['id'])
        qa_datum.set_question(val['question'])
        qa_datum.set_context(val['context'])
        qa_datum.set_answers(QAAnswer.from_dict(i) for i in (val['answers']))
        return qa_datum
