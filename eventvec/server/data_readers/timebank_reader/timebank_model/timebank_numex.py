class TimebankNumex:
    def __init__(self):
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
    def from_bs_obj(bs_obj):
        timebank_numex = TimebankNumex()
        timebank_numex.set_type(bs_obj.attrs['type'])
        timebank_numex.set_text(bs_obj.text)
        return timebank_numex

    def to_dict(self):
        return {
            'object_type': 'timebank_numex',
            'type': self.type(),
            'text': self.text(),
        }
