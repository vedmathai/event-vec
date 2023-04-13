import json


class Config:
    _instance = None

    def __init__(self):
        self._matres_data_location = None
        self._experiment_type = None
        self._book_corpus_data_location = None
        self._heatmaps_location = None
        self._tsqa_data_location = None
        self._tsqa_file_names = None
        self._tsqa_file2annotation_map = {}
        self._torque_data_location = None
        self._torque_data_file_names = None


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

    def run_configs_file(self):
        return self._run_configs_file

    def experiment_type(self):
        return self._experiment_type

    def book_corpus_data_location(self):
        return self._book_corpus_data_location

    def model_save_location(self):
        return self._model_save_location

    def heatmaps_location(self):
        return self._heatmaps_location

    def tsqa_data_location(self):
        return self._tsqa_data_location

    def tsqa_file_names(self):
        return self._tsqa_file_names

    def tsqa_file2annotation_map(self):
        return self._tsqa_file2annotation_map

    def torque_data_location(self):
        return self._torque_data_location

    def torque_data_file_names(self):
        return self._torque_data_file_names

    def set_timebank_data_location(self, timebank_data_location):
        self._timebank_data_location = timebank_data_location

    def timebank_dense_data_location(self):
        return self._timebank_dense_data_location

    def set_timebank_dense_data_location(self, timebank_dense_data_location):
        self._timebank_dense_data_location = timebank_dense_data_location

    def set_run_configs_file(self, run_configs_file):
        self._run_configs_file = run_configs_file

    def set_experiment_type(self, experiment_type):
        self._experiment_type = experiment_type

    def set_book_corpus_data_location(self, book_corpus_data_location):
        self._book_corpus_data_location = book_corpus_data_location

    def set_model_save_location(self, model_save_location):
        self._model_save_location = model_save_location

    def set_heatmaps_location(self, heatmaps_location):
        self._heatmaps_location = heatmaps_location

    def set_tsqa_data_location(self, tsqa_data_location):
        self._tsqa_data_location = tsqa_data_location

    def set_tsqa_file_names(self, tsqa_file_names):
        self._tsqa_file_names = tsqa_file_names

    def set_tsqa_file2annotation_map(self, tsqa_file2annotation_map):
        self._tsqa_file2annotation_map = tsqa_file2annotation_map

    def set_torque_data_location(self, torque_data_location):
        self._torque_data_location = torque_data_location

    def set_torque_data_file_names(self, torque_data_file_names):
        self._torque_data_file_names = torque_data_file_names

    @staticmethod
    def from_dict(val):
        config = Config()
        config.set_matres_data_location(val.get('matres_data_location'))
        config.set_timebank_data_location(val.get('timebank_data_location'))
        config.set_timebank_dense_data_location(val.get('timebank_dense_data_location'))
        config.set_book_corpus_data_location(val.get('book_corpus_data_location'))
        config.set_experiment_type(val.get('experiment_type'))
        config.set_run_configs_file(val.get('run_configs_file'))
        config.set_model_save_location(val.get('model_save_location'))
        config.set_heatmaps_location(val.get('heatmaps_location'))
        config.set_tsqa_data_location(val.get('tsqa_data_location'))
        config.set_tsqa_file_names(val.get('tsqa_file_names'))
        config.set_tsqa_file2annotation_map(val.get('tsqa_file2annotation_map'))
        config.set_torque_data_location(val.get('torque_data_location'))
        config.set_torque_data_file_names(val.get('torque_data_file_names'))
        return config
