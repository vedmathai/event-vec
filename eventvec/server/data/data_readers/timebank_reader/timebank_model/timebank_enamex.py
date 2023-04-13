from eventvec.server.data_readers.timebank_reader.timebank_model.timebank_abstract_sp import AbstractSentencePart  # noqa


class TimebankEnamex(AbstractSentencePart):
    def __init__(self):
        super().__init__()
        self._type = None
        self._text = None

    def type(self):
        return self._type

    def text(self):
        return self._text

    def set_type(self, type):
        self._type = type

    def set_text(self, text):
        self._text = text

    @staticmethod
    def from_bs_obj(bs_obj, token_i, last_token):
        timebank_enamex = TimebankEnamex()
        timebank_enamex.set_start_token_i(last_token + 1)
        end_token = last_token + len(bs_obj.text.split())
        timebank_enamex.set_end_token_i(end_token)
        timebank_enamex.set_sentence_token_i(token_i)
        timebank_enamex.set_sentence_token_i(token_i)
        timebank_enamex.set_type(bs_obj.attrs.get('type'))
        timebank_enamex.set_text(bs_obj.text)
        return timebank_enamex, end_token

    def to_dict(self):
        return {
            'object_type': 'timebank_enamex',
            'type': self.type(),
            'text': self.text(),
        }
