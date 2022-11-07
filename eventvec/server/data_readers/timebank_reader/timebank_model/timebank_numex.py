from eventvec.server.data_readers.timebank_reader.timebank_model.timebank_abstract_sp import AbstractSentencePart  # noqa


class TimebankNumex(AbstractSentencePart):
    def __init__(self):
        super().__init__()
        self._sentence_token_i = None
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
        timebank_numex = TimebankNumex()
        timebank_numex.set_start_token_i(last_token + 1)
        end_token = last_token + len(bs_obj.text.split())
        timebank_numex.set_end_token_i(end_token)
        timebank_numex.set_sentence_token_i(token_i)
        timebank_numex.set_type(bs_obj.attrs['type'])
        timebank_numex.set_text(bs_obj.text)
        return timebank_numex, end_token

    def to_dict(self):
        return {
            'object_type': 'timebank_numex',
            'type': self.type(),
            'text': self.text(),
        }
