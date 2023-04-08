from eventvec.server.data_readers.torque_reader.torque_model.torque_datum import TorqueDatum


class TorqueDataset:
    def __init__(self):
        self._data = []

    def data(self):
        return self._data

    def set_data(self, data):
        self._data = data

    def extend_data(self, data):
        self._data.extend(data)

    @staticmethod
    def from_dict(val):
        dataset = TorqueDataset()
        for item in val:
            passage_data = [TorqueDatum.from_dict(i) for i in item['passages']]
            dataset.extend_data(passage_data)
        return dataset

    def to_dict(self):
        return {
            k: v.to_dict() for k, v in self.data()
        }