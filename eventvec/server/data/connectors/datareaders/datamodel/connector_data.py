class ConnectorData:
    def __init__(self):
        self._data = []

    def set_data(self, data):
        self._data = data

    def data(self):
        return self._data
    
    def add_datum(self, datum):
        self._data.append(datum)