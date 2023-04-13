from eventvec.server.reporters.report_model.epoch_statistics import EpochStatistics  # noqa


class ReportModel:
    def __init__(self):
        self._current_epoch_idx = -1
        self._epochs = []
        self._labels = []

    def register_new_epoch(self):
        self._current_epoch_idx += 1
        self._epochs.append(EpochStatistics())
        self.current_epoch().set_labels(self._labels)

    def current_epoch(self):
        return self._epochs[self._current_epoch_idx]

    def epochs(self):
        return self._epochs

    def set_labels(self, labels):
        self._labels = list(labels)