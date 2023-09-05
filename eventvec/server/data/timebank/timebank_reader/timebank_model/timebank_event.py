from eventvec.server.data.timebank.timebank_reader.timebank_model.timebank_abstract_sp import AbstractSentencePart  # noqa


class TimebankEvent(AbstractSentencePart):
    def __init__(self):
        super().__init__()
        self._sentence_token_i = None
        self._eid = None
        self._class = None
        self._text = None

    def eid(self):
        return self._eid

    def event_class(self):
        return self._event_class

    def text(self):
        return self._text

    def set_eid(self, eid):
        self._eid = eid

    def set_event_class(self, event_class):
        self._event_class = event_class

    def set_text(self, text):
        self._text = text

    @staticmethod
    def from_bs_obj(bs_obj, token_i, last_token):
        timebank_event = TimebankEvent()
        timebank_event.set_start_token_i(last_token + 1)
        end_token = last_token + len(bs_obj.text.split())
        timebank_event.set_end_token_i(end_token)
        timebank_event.set_sentence_token_i(token_i)
        timebank_event.set_eid(bs_obj.attrs['eid'])
        timebank_event.set_event_class(bs_obj.attrs['class'])
        timebank_event.set_text(bs_obj.text)
        return timebank_event, end_token

    def to_dict(self):
        return {
            'eid': self.eid(),
            'class': self.event_class(),
            'text': self.text(),
        }
