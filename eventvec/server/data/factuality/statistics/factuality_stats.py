from collections import defaultdict
import numpy as np

from eventvec.server.data.factuality.factuality_readers.factuality_reader import FactualityReader  # noqa
from eventvec.server.featurizers.factuality_categorizer.factuality_categorizer import FactualityCategorizer



class FactualityStats():
    def __init__(self):
        self._factuality_reader = FactualityReader()
        self._factuality_categorizer = FactualityCategorizer()


    def calculate(self):
        data = self._factuality_reader.belief_data().data()
        counter = defaultdict(list)
        worker2score = defaultdict(list)
        for datumi, datum in enumerate(data):
            annotations = []
            for annotation in datum.annotations():
                worker2score[annotation.worker_id()] += [annotation.value()]
                annotations.append(annotation.value())
            mean = np.mean(annotations)
            std = np.std(annotations)
            if datum.event_string() == 'cancellations':
                print(datum.text())
            features_array = self._factuality_categorizer.categorize(datum.text(), datum.event_string())
            features_array = features_array.to_dict()
            for feature in features_array:
                if features_array[feature] is True:
                    counter[feature].append(mean)
            if all(i is False for i in features_array.values()):
                counter['no_feature'].append(mean)
        for key in counter:
            print(key, np.mean(counter[key]), np.std(counter[key]))

if __name__ == '__main__':
    stats = FactualityStats()
    stats.calculate()