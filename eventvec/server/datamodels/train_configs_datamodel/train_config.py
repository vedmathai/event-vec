class TrainConfig():
    def __init__(self):
        self._model_type = None
        self._llm = None

    def model_type(self):
        return self._model_type

    def llm(self):
        return self._llm

    def set_model_type(self, model_type):
        self._model_type = model_type

    def set_llm(self, llm):
        self._llm = llm

    @staticmethod
    def from_dict(train_config_dict):
        train_config = TrainConfig()
        train_config.set_llm(train_config_dict['llm'])
        train_config.set_model_type(train_config_dict['model_type'])
        return train_config

    def to_dict(self):
        return {
            'llm': self.llm(),
            'model_type': self.model_type()
        }
