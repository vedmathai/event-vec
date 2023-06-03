from eventvec.server.datamodels.qa_datamodels.qa_answer import QAAnswer


class QADatum:
    def __init__(self):
        self._id = None
        self._question = None
        self._context = []
        self._answers = []
        self._alternate_answer_sets = []
        self._question_events = None
        self._use_in_eval = False
        self._context_events = []

    def id(self):
        return self._id

    def question(self):
        return self._question

    def context(self):
        return self._context

    def answers(self):
        return self._answers
    
    def question_events(self):
        return self._question_events
    
    def alternate_answer_sets(self):
        return self._alternate_answer_sets
    
    def use_in_eval(self):
        return self._use_in_eval
    
    def context_events(self):
        return self._context_events

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

    def set_alternate_answer_sets(self, alternate_answer_sets):
        self._alternate_answer_sets = alternate_answer_sets

    def add_alternate_answer_set(self, alternate_answer_set):
        self._alternate_answer_sets.append(alternate_answer_set)

    def set_question_events(self, question_events):
        self._question_events = question_events

    def set_use_in_eval(self, use_in_eval):
        self._use_in_eval = use_in_eval

    def set_context_events(self, context_events):
        self._context_events = context_events

    def add_context_event(self, context_event):
        self._context_events.append(context_event)

    def to_dict(self):
        return {
            "id": self.id(),
            "question": self.question(),
            "context": self.context(),
            "answers": [i.to_dict() for i in self.answers()],
            "alternate_answer_sets": [[i.to_dict() for i in k] for k in self.alternate_answer_sets()],
            "question_events": self.question_events(),
            "use_in_eval": self.use_in_eval(),
            "context_events": self.context_events(),
        }

    @staticmethod
    def from_dict(val):
        qa_datum = QADatum()
        qa_datum.set_id(val['id'])
        qa_datum.set_question(val['question'])
        qa_datum.set_context(val['context'])
        qa_datum.set_answers(QAAnswer.from_dict(i) for i in (val['answers']))
        qa_datum.set_alternate_answer_sets([[QAAnswer.from_dict(i) for i in k] for k in val['alternate_answer_sets']])
        qa_datum.set_question_events(val['question_events'])
        qa_datum.set_use_in_eval(val['use_in_eval'])
        qa_datum.set_context_events(val['context_events'])
        return qa_datum
