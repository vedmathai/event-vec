

class TorqueEventsAnswer:
    def __init__(self):
        self._indices = None
        self._spans = None

    def indices(self):
        return self._indices

    def spans(self):
        return self._spans
    
    def set_indices(self, indices):
        self._indices = indices

    def set_spans(self, spans):
        self._spans = spans

    @staticmethod
    def from_dict(val):
        datum = TorqueEventsAnswer()
        datum.set_indices([eval(i) for i in val['indices']])
        datum.set_spans(val['spans'])
        return datum

    def to_dict(self):
        return {
            "indices": self.indices(),
            "spans": self.spans(),
        }
