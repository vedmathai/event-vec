from eventvec.server.data_readers.timebank_reader.timebank_model.timebank_abstract_sp import AbstractSentencePart  # noqa


class TimebankSignal(AbstractSentencePart):
    def __init__(self):
        super().__init__()
        self._sid = None
        self._text = None

    def sid(self):
        return self._sid

    def text(self):
        return self._text

    def set_sid(self, sid):
        self._sid = sid

    def set_text(self, text):
        self._text = text

    @staticmethod
    def from_bs_obj(bs_obj, token_i, last_token):
        timebank_signal = TimebankSignal()
        timebank_signal.set_start_token_i(last_token + 1)
        end_token = last_token + len(bs_obj.text.split())
        timebank_signal.set_end_token_i(end_token)
        timebank_signal.set_sentence_token_i(token_i)
        timebank_signal.set_sid(bs_obj.attrs['sid'])
        timebank_signal.set_text(bs_obj.text)
        return timebank_signal, end_token

    def to_dict(self):
        return {
            'object_type': 'timebank_signal',
            'sid': self.sid(),
            'text': self.text(),
        }
