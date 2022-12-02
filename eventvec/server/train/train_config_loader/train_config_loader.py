import json

from eventvec.server.config import Config
from eventvec.server.model.train_configs.train_configs import TrainConfigs


class TrainConfigsLoader:
    def __init__(self):
        self._config = Config.instance()
        self._train_configs = None

    def load(self):
        with open(self._config.train_configs_file(), 'rt') as f:
            train_configs_dict = json.load(f)
            self._train_configs = TrainConfigs.from_dict(train_configs_dict)
        return train_configs_dict

    def train_configs(self):
        return self._train_configs
