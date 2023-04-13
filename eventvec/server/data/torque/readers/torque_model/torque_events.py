from eventvec.server.data.torque.readers.torque_model.torque_events_answer import TorqueEventsAnswer


class TorqueEvents:
    def __init__(self):
        self._answer = None
        self._passage_id = None

    def answer(self):
        return self._answer

    def passage_id(self):
        return self._passage_id

    def set_answer(self, answer):
        self._answer = answer

    def set_passage_id(self, passage_id):
        self._passage_id = passage_id

    @staticmethod
    def from_dict(val):
        datum = TorqueEvents()
        datum.set_answer(TorqueEventsAnswer.from_dict(val[0]['answer']))
        datum.set_passage_id(val[0]['passageID'])
        return datum
    
    @staticmethod
    def from_eval_dict(val):
        datum = TorqueEvents()
        datum.set_answer(TorqueEventsAnswer.from_dict(val['answer']))
        datum.set_passage_id(val['passageID'])
        return datum

    def to_dict(self):
        return {
            "answer": self.answer().to_dict(),
            "passage_id": self.passage_id(),
        }
