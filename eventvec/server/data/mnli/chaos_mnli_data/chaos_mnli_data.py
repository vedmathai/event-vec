from eventvec.server.data.mnli.chaos_mnli_data.chaos_mnli_datum import ChaosMNLIDatum


class ChaosMNLIData:
    def __init__(self):
        self._data = []

    def add_datum(self, datum):
        self._data.append(datum)

    def data(self):
        return self._data
    
    def set_data(self, data):
        self._data = data

    @staticmethod
    def from_jsonl(jsonl):
        data = ChaosMNLIData()
        for line in jsonl:
            datum = ChaosMNLIDatum.from_json(line)
            data.add_datum(datum)
        return data