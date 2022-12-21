from collections import defaultdict
import numpy as np


from eventvec.server.reporter.heatmaps.heatmap_report import HeatmapReport


class RunStatistics:
    def __init__(self, run_name):
        self._run_name = run_name
        self._iterations = 0
        self._total_loss = 0
        self._true_positives = defaultdict(int)
        self._false_positives = defaultdict(int)
        self._false_negatives = defaultdict(int)
        self._confusion_matrix = defaultdict(lambda: defaultdict(int))
        self._counts = defaultdict(int)
        self._heatmap_report = HeatmapReport()
        self._labels = []

    def set_labels(self, labels):
        self._labels = list(set(labels))

    def labels(self):
        return self._labels

    def record_iteration(self, predicted, expected, loss):
        self._iterations += 1
        self._counts[expected] += 1
        if predicted == expected:
            self._true_positives[predicted] += 1
        else:
            if predicted != expected:
                self._false_positives[predicted] += 1
                self._false_negatives[expected] += 1
        self._confusion_matrix[predicted][expected] += 1
        self._total_loss += loss

    def iterations(self):
        return self._iterations

    def true_positives(self, label):
        return self._true_positives[label]

    def false_positives(self, label):
        return self._false_positives[label]

    def false_negatives(self, label):
        return self._false_negatives[label]

    def precision(self, label):
        numerator = float(self.true_positives(label))
        denominator = self.true_positives(label) + self.false_positives(label)
        if denominator == 0:
            return 0
        return numerator/denominator

    def precisions(self):
        precisions = {}
        for label in self.labels():
            precisions[label] = self.precision(label)
        return precisions

    def recall(self, label):
        numerator = float(self.true_positives(label))
        denominator = self.true_positives(label) + self.false_negatives(label)
        if denominator == 0:
            return 0
        return numerator/denominator

    def recalls(self):
        recalls = {}
        for label in self.labels():
            recalls[label] = self.recall(label)
        return recalls

    def f1(self, label):
        precision = self.precision(label)
        recall = self.recall(label)
        numerator = 2 * precision * recall
        denominator = precision + recall
        if denominator == 0:
            return 0
        return numerator / denominator

    def f1s(self):
        f1s = {}
        for label in self.labels():
            f1s[label] = self.f1(label)
        return f1s

    def macro_f1(self):
        f1s = []
        for label in self.labels():
            f1s.append(self.f1(label))
        return np.mean(f1s)

    def weighted_macro_f1(self):
        f1s = []
        total_counts = sum(self._counts.values())
        for label in self.labels():
            weight = self._counts[label] / total_counts
            f1s.append(self.f1(label) * weight)
        return np.mean(f1s)

    def mean_loss(self):
        return self._total_loss / self._iterations

    def accuracy(self):
        numerator = sum(self._true_positives.values())
        denominator = float(self._iterations)
        return numerator / denominator

    def generate_confusion_heatmaps(self):
        self._heatmap_report.generate_heatmap(
            self._confusion_matrix, self.labels(), self._run_name
        )

    def to_dict(self):
        return {
            'precision': self.precisions(),
            'recall': self.recalls(),
            'f1': self.f1s(),
            'mean_loss': self.mean_loss(),
            'accuracy': self.accuracy(),
            'macro_f1': self.macro_f1(),
            'weighted_macro_f1': self.weighted_macro_f1(),
        }
