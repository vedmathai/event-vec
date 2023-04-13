from eventvec.server.data_readers.torque_reader.torque_model.torque_events import TorqueEvents
from eventvec.server.data_readers.torque_reader.torque_model.torque_question_answer_pairs import TorqueQuestionAnswerPairs


class TorqueDatum:
    def __init__(self):
        self._events = None
        self._passage = None
        self._question_answer_pairs = None

    def events(self):
        return self._events

    def passage(self):
        return self._passage

    def question_answer_pairs(self):
        return self._question_answer_pairs

    def set_events(self, events):
        self._events = events
    
    def set_passage(self, passage):
        self._passage = passage

    def set_question_answer_pairs(self, question_answer_pairs):
        self._question_answer_pairs = question_answer_pairs

    @staticmethod
    def from_train_dict(val):
        datum = TorqueDatum()
        datum.set_passage(val['passage'])
        datum.set_events(TorqueEvents.from_dict(val['events']))
        datum.set_question_answer_pairs(TorqueQuestionAnswerPairs.from_dict(val['question_answer_pairs']))
        return datum
    
    @staticmethod
    def from_eval_dict(val):
        datum = TorqueDatum()
        datum.set_passage(val['passage'])
        datum.set_events(TorqueEvents.from_eval_dict(val['events']))
        datum.set_question_answer_pairs(TorqueQuestionAnswerPairs.from_eval_dict(val['question_answer_pairs'].items()))
        return datum

    def to_dict(self):
        return {
            "events": self.events().to_dict(),
            "passage": self.passage(),
            "question_answer_pairs": self.question_answer_pairs.to_dict(),
        }
