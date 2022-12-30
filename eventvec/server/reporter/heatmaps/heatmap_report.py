import matplotlib.pyplot as plt
import numpy as np
import os

from eventvec.server.config import Config


class HeatmapReport:
    def __init__(self):
        config = Config.instance()
        self._heatmaps_location = config.heatmaps_location()

    def generate_heatmap(self, data_dict, labels, name):
        num_nodes = len(labels)
        matrix = np.zeros((num_nodes, num_nodes))
        for label_x in labels:
            for label_y in labels:
                matrix[label_x, label_y] = data_dict[label_x][label_y]
        self.save_heatmap(matrix, name)

    def save_heatmap(self, matrix, name):
        filename = os.path.join(self._heatmaps_location, name)
        plt.imshow(matrix, cmap='hot', interpolation='nearest')
        plt.savefig(filename)
