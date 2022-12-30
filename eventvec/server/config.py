import json


class Config:
    _instance = None

    def __init__(self):
        self._matres_data_location = None
        self._experiment_type = None
        self._book_corpus_data_location = None
        self._heatmaps_location = None

    @staticmethod
    def instance():
        if Config._instance is None:
            with open('eventvec/server/config.json') as f:
                Config._instance = Config.from_dict(json.load(f))
        return Config._instance

    def matres_data_location(self):
        return self._matres_data_location

    def set_matres_data_location(self, matres_data_location):
        self._matres_data_location = matres_data_location

    def timebank_data_location(self):
        return self._timebank_data_location

    def train_configs_file(self):
        return self._train_configs_file

    def experiment_type(self):
        return self._experiment_type

    def book_corpus_data_location(self):
        return self._book_corpus_data_location

    def model_save_location(self):
        return self._model_save_location

    def heatmaps_location(self):
        return self._heatmaps_location

    def set_timebank_data_location(self, timebank_data_location):
        self._timebank_data_location = timebank_data_location

    def timebank_dense_data_location(self):
        return self._timebank_dense_data_location

    def set_timebank_dense_data_location(self, timebank_dense_data_location):
        self._timebank_dense_data_location = timebank_dense_data_location

    def set_train_configs_file(self, train_configs_file):
        self._train_configs_file = train_configs_file

    def set_experiment_type(self, experiment_type):
        self._experiment_type = experiment_type

    def set_book_corpus_data_location(self, book_corpus_data_location):
        self._book_corpus_data_location = book_corpus_data_location

    def set_model_save_location(self, model_save_location):
        self._model_save_location = model_save_location

    def set_heatmaps_location(self, heatmaps_location):
        self._heatmaps_location = heatmaps_location

    @staticmethod
    def from_dict(val):
        config = Config()
        config.set_matres_data_location(val.get('matres_data_location'))
        config.set_timebank_data_location(val.get('timebank_data_location'))
        config.set_timebank_dense_data_location(val.get('timebank_dense_data_location'))
        config.set_book_corpus_data_location(val.get('book_corpus_data_location'))
        config.set_experiment_type(val.get('experiment_type'))
        config.set_train_configs_file(val.get('train_configs_file'))
        config.set_model_save_location(val.get('model_save_location'))
        config.set_heatmaps_location(val.get('heatmaps_location'))
        return config
