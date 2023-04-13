from eventvec.server.data_readers.timebank_reader.timebank_model.timebank_abstract_sp import AbstractSentencePart  # noqa


class TimebankTimex(AbstractSentencePart):
    def __init__(self):
        super().__init__()
        self._sentence_token_i = None
        self._tid = None
        self._type = None
        self._value = None
        self._temporal_function = None
        self._function_in_document = None
        self._anchor_time_id = None
        self._text = None

    def tid(self):
        return self._tid

    def type(self):
        return self._type

    def value(self):
        return self._value

    def temporal_function(self):
        return self._temporal_function

    def function_in_document(self):
        return self._function_in_document

    def anchor_time_id(self):
        return self._anchor_time_id

    def text(self):
        return self._text

    def set_tid(self, tid):
        self._tid = tid

    def set_type(self, type):
        self._type = type

    def set_value(self, value):
        self._value = value

    def set_temporal_function(self, temporal_function):
        self._temporal_function = temporal_function

    def set_function_in_document(self, function_in_document):
        self._function_in_document = function_in_document

    def set_anchor_time_id(self, anchor_time_id):
        self._anchor_time_id = anchor_time_id

    def set_text(self, text):
        self._text = text

    @staticmethod
    def from_bs_obj(timex, token_i, last_token):
        timebank_timex = TimebankTimex()
        timebank_timex.set_start_token_i(last_token + 1)
        end_token = last_token + len(timex.text.split())
        timebank_timex.set_end_token_i(end_token)
        timebank_timex.set_sentence_token_i(token_i)
        timebank_timex.set_tid(timex.attrs.get('tid'))
        timebank_timex.set_type(timex.attrs.get('type'))
        timebank_timex.set_value(timex.attrs.get('value'))
        timebank_timex.set_temporal_function(timex.attrs.get('temporal_function'))  # noqa
        timebank_timex.set_function_in_document(timex.attrs.get('function_id_document'))  # noqa
        timebank_timex.set_anchor_time_id(timex.attrs.get('anchor_time_id'))
        timebank_timex.set_text(timex.text)
        return timebank_timex, end_token

    def to_dict(self):
        return {
            'object_type': 'timebank_timex',
            'text': self.text(),
            'anchor_time_id': self.anchor_time_id(),
            'function_in_document': self.function_in_document(),
            'temporal_function': self.temporal_function(),
            'value': self.value(),
            'type': self.type(),
            'tid': self.tid(),
        }
