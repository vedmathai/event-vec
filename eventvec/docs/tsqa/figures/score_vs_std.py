

import numpy as np  
import matplotlib.pyplot as plt  


from collections import defaultdict
from eventvec.server.data.factuality.factuality_readers.factuality_reader import FactualityReader  # noqa
from eventvec.server.featurizers.factuality_categorizer.factuality_categorizer import FactualityCategorizer



class PlotFactuality():
    def __init__(self):
        self._factuality_reader = FactualityReader()
        self._factuality_categorizer = FactualityCategorizer()


    def plot(self):
        data = self._factuality_reader.belief_data().data()
        counter = defaultdict(int)
        X = []
        Y = []
        for datum in data:
            annotations = []
            for annotation in datum.annotations():
                annotations.append(annotation.value())
            mean = np.mean(annotations)
            std = np.std(annotations)
            std = int(std)
            counter[int(std)] += 1
            Y.append(mean)
            X.append(std)
        print(counter)

        plt.scatter(X, Y, alpha=0.02)
        plt.xlabel("Standard Deviation of Annotations") 
        plt.ylabel("Mean of Annotations") 
        plt.title("Plotting the mean and standard deviation of\n factuality annotations in the belief data rounded down.") 
        plt.show()


if __name__ == '__main__':
    stats = PlotFactuality()
    stats.plot()
  
#plt.xticks(X_axis, X) 
