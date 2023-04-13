from eventvec.server.model.train_configs.train_config import TrainConfig


class TrainConfigs:
    def __init__(self):
        self._train_configs = []

    def train_configs(self):
        return self._train_configs

    def set_train_configs(self, train_configs):
        self._train_configs = train_configs

    def add_train_config(self, train_config):
        self._train_configs.append(train_config)

    @staticmethod
    def from_dict(train_configs_dict):
        train_configs = TrainConfigs()
        for train_config_dict in train_configs_dict['train_configs']:
            train_config = TrainConfig.from_dict(train_config_dict)
            train_configs.add_train_config(train_config)
        return train_configs

    def to_dict(self):
        return {
            'train_configs': [i.to_dict() for i in self._train_configs()]
        }
