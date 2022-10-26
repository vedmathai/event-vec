
class TimebankTextSegment:
    def __init__(self):
        self._text = None

    def text(self):
        return self._text

    def set_text(self, text):
        self._text = text

    @staticmethod
    def from_bs_obj(text):
        timebank_text_segment = TimebankTextSegment()
        timebank_text_segment.set_text(text.text.strip())
        return timebank_text_segment

    def to_dict(self):
        return {
            'object_type': 'timebank_text_segment',
            'text': self.text(),
        }
