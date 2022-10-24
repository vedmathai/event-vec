class TimebankEnamex:
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
        timebank_enamex = TimebankEnamex()
        timebank_enamex.set_type(bs_obj.attrs.get('type'))
        timebank_enamex.set_text(bs_obj.text)
        return timebank_enamex

    def to_dict(self):
        return {
            'object_type': 'timebank_enamex',
            'type': self.type(),
            'text': self.text(),
        }
