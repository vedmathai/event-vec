from eventvec.server.datamodels.qa_datamodels.qa_datum import QADatum


class QADataset:
    def __init__(self):
        self._name = None
        self._data = []

    def name(self):
        return self._name

    def data(self):
        return self._data

    def set_name(self, name):
        self._name = name

    def set_data(self, data):
        self._data = data

    def add_datum(self, datum):
        self._data.append(datum)

    def to_dict(self):
        return {
            "name": self.name(),
            "data": [i.to_dict() for i in self.data()],
        }

    @staticmethod
    def from_dict(val):
        qa_datum = QADataset()
        qa_datum.set_name(val['name'])
        qa_datum.set_data(QADatum.from_dict(i) for i in val['data'])
        return qa_datum
