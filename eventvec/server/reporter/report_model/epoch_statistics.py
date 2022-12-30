from eventvec.server.reporter.report_model.run_statistics import RunStatistics


class EpochStatistics:
    def __init__(self):
        self._epoch_number = 0
        self._labels = []
        self._train_statistics = RunStatistics('train')
        self._test_statistics = RunStatistics('test')

    def epoch_number(self):
        return self._epoch_number

    def train_statistics(self):
        return self._train_statistics

    def test_statistics(self):
        return self._test_statistics

    def set_labels(self, labels):
        self._train_statistics.set_labels(labels)
        self._test_statistics.set_labels(labels)

    def set_epoch_number(self, epoch_number):
        self._epoch_number = epoch_number

    def set_train_stats(self, train_statistics):
        self._train_statistics = train_statistics

    def set_test_statistics(self, test_statistics):
        self._test_statistics = test_statistics

    def record_train_iteration(self, predicted, expected, loss):
        self._train_statistics.record_iteration(predicted, expected, loss)

    def record_test_iteration(self, predicted, expected, loss):
        self._test_statistics.record_iteration(predicted, expected, loss)

    def generate_confusion_heatmaps(self):
        self._train_statistics.generate_confusion_heatmaps()
        self._test_statistics.generate_confusion_heatmaps()

    def to_dict(self):
        return {
            'train_statistics': self._train_statistics.to_dict(),
            'test_statistics': self._test_statistics.to_dict(),
        }
