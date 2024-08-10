class ROCData():
    def __init__(self):
        self._data = []

    def add_datum(self, datum):
        self._data.append(datum)

    def data(self):
        return self._data
    
    def to_dict(self):
        return {
            'data': [datum.to_dict() for datum in self._data],
        }
