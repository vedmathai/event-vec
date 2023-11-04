from typing import Any, Dict

from eventvec.server.data.factuality.factuality_datamodels.belief_datum import BeliefDatum


class BeliefData():
    def __init__(self):
        super().__init__()
        self._data = []

    def data(self):
        return self._data
    
    def set_data(self, data):
        self._data = data

    def add_datum(self, datum):
        self._data.append(datum)

    def to_dict(self):
        return {
            'data': [d.to_dict() for d in self.data()]
        }
    
    @staticmethod
    def from_dict(dict_data: Dict[str, Any]):
        belief_data = BeliefData()
        for datum in dict_data['data']:
            belief_datum = BeliefDatum.from_dict(datum)
            belief_data.add_datum(belief_datum)
        return belief_data
