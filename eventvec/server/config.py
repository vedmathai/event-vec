import json


class Config:
    _instance = None

    def __init__(self):
        self._matres_data_location = None

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

    def set_timebank_data_location(self, timebank_data_location):
        self._timebank_data_location = timebank_data_location

    @staticmethod
    def from_dict(val):
        config = Config()
        config.set_matres_data_location(val.get('matres_data_location'))
        config.set_timebank_data_location(val.get('timebank_data_location'))
        return config
