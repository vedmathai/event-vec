class TimebankSignal:
    def __init__(self):
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
    def from_bs_obj(bs_obj):
        timebank_signal = TimebankSignal()
        timebank_signal.set_sid(bs_obj.attrs['sid'])
        timebank_signal.set_text(bs_obj.text)
        return timebank_signal

    def to_dict(self):
        return {
            'object_type': 'timebank_signal',
            'sid': self.sid(),
            'text': self.text(),
        }
