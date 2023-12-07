

class Annotation():
    def __init__(self):
        super().__init__()
        self._value = None

    def set_value(self, value):
        self._value = value
    
    def value(self):
        return self._value
    
    @staticmethod
    def from_dict(dict_data):
        annotation = Annotation()
        annotation.set_value(dict_data['value'])
        return annotation
    
    def to_json(self):
        return {
            'value': self.value()
        }
    
    @staticmethod
    def parse_raw_data(raw_data):
        annotation = Annotation()
        annotation.set_value(int(raw_data['data']['belief_question']))
        return annotation
