from eventvec.server.data.timebank.timebank_reader.timebank_model.timebank_abstract_sp import AbstractSentencePart  # noqa


class TimebankTextSegment(AbstractSentencePart):
    def __init__(self):
        super().__init__()
        self._sentence_token_i = None
        self._text = None

    def text(self):
        return self._text

    def set_text(self, text):
        self._text = text

    @staticmethod
    def from_bs_obj(text, token_i, last_token):
        timebank_text_segment = TimebankTextSegment()
        timebank_text_segment.set_start_token_i(last_token + 1)
        end_token = last_token + len(text.text.strip().split())
        timebank_text_segment.set_end_token_i(end_token)
        timebank_text_segment.set_sentence_token_i(token_i)
        timebank_text_segment.set_sentence_token_i(token_i)
        timebank_text_segment.set_text(text.text.strip())
        return timebank_text_segment, end_token

    def to_dict(self):
        return {
            'object_type': 'timebank_text_segment',
            'text': self.text(),
        }
