from typing import Dict
import csv
import csv

from eventvec.server.data.connectors.datareaders.datamodel.connector_data import ConnectorData
from eventvec.server.data.connectors.datareaders.datamodel.connector_datum import ConnectorDatum
from eventvec.server.data.abstract import AbstractDatareader


class ConnectorsDatareader(AbstractDatareader):
    def __init__(self):
        super().__init__()
        self.folder = self._config.connectors_data_location()

    def read_file(self, filepath):
        with open(filepath) as f:
            return f.read()

    def data(self) -> Dict:
        filepath = self.folder
        with open(filepath) as f:
            reader = csv.reader(f, delimiter='\t')
            data = ConnectorData()
            for r in reader:
                datum = ConnectorDatum()
                datum.set_uid(str(r[0]))
                datum.set_para(r[2])
                datum.set_label(r[1])
                data.add_datum(datum)
        return data
    

if __name__ == '__main__':
    reader = ConnectorsDatareader()
    d = reader.data()
    for r in d.data():
        print(type(r.label()))