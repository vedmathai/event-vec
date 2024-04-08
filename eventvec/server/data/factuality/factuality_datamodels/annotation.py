

class Annotation():
    def __init__(self):
        super().__init__()
        self._value = None
        self._worker_id = None

    def set_value(self, value):
        self._value = value
    
    def value(self):
        return self._value
    
    def set_worker_id(self, worker_id):
        self._worker_id = worker_id

    def worker_id(self):
        return self._worker_id
    
    @staticmethod
    def from_dict(dict_data):
        annotation = Annotation()
        annotation.set_value(dict_data['value'])
        annotation.set_worker_id(dict_data['worker_id'])
        return annotation
    
    def to_json(self):
        return {
            'value': self.value(),
            'worker_id': self.worker_id(),
        }
    
    @staticmethod
    def parse_raw_data(raw_data):
        annotation = Annotation()
        annotation.set_value(int(raw_data['data']['belief_question']))
        annotation.set_worker_id(raw_data['worker_id'])
        return annotation
