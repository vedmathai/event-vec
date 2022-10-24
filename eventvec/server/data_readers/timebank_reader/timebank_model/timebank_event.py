
class TimebankEvent:
    def __init__(self):
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
    def from_bs_obj(bs_obj):
        timebank_event = TimebankEvent()
        timebank_event.set_eid(bs_obj.attrs['eid'])
        timebank_event.set_event_class(bs_obj.attrs['class'])
        timebank_event.set_text(bs_obj.text)
        return timebank_event

    def to_dict(self):
        return {
            'eid': self.eid(),
            'class': self.event_class(),
            'text': self.text(),
        }
