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
        for datum in data:
            annotations = []
            for annotation in datum.annotations():
                worker2score[annotation.worker_id()] += [annotation.value()]
                annotations.append(annotation.value())
            mean = np.mean(annotations)
            features_array = self._factuality_categorizer.categorize(datum.text(), datum.event_string())
            total = 0
            for k, v in  features_array.to_dict().items():
                if v is True:
                    total += 1
            for k, v in  features_array.to_dict().items():
                if v is True and total == 1:
                    counter[k] += [mean]
                if total == 0:
                    counter['no_feature'] += [mean]
        print({k: np.mean(v) for k, v in counter.items()})
        print(worker2score)

        worker2set = defaultdict(int)
        for k, v in worker2score.items():
            if len(set(v)) == 4:
                print(worker2score[k])
            worker2set[k] = len(set(v))


        score2number = defaultdict(int)
        for k, v in worker2set.items():
            score2number[v] += 1
        
        print(score2number)


if __name__ == '__main__':
    stats = FactualityStats()
    stats.calculate()