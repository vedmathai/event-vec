from collections import defaultdict
import csv

from eventvec.server.featurizers.lingusitic_featurizer import LinguisticFeaturizer
from eventvec.server.data.wikipedia.datahandlers.wiki_datahandler import WikiDatahandler


class ConnectorNLICellCreator():
    def __init__(self):
        self._counter = {'but': 0, 'and': 0, 'because': 0, 'so': 0, 'though': 0}
        self._dataset = []

    def create_connector_dataset(self):
        cells = []
        counter = 0
        for connector_premise in ['and', 'because', 'so', 'though', 'but']:
            for connector_hypothesis in ['and', 'because', 'so', 'though', 'but']:
                for switch in ['straight', 'switch']:
                    for count in range(20):
                        counter += 1
                        self._dataset.append([counter, connector_premise, connector_hypothesis, switch])
        self.save_dataset()

    def save_dataset(self):
        with open('/home/lalady6977/oerc/projects/data/connector_nli_cells2.csv', 'wt') as f:
            writer = csv.writer(f, delimiter='\t')
            for row in self._dataset:
                writer.writerow(row)


if __name__ == '__main__':
    creator = ConnectorNLICellCreator()
    creator.create_connector_dataset()
    print(creator._counter)