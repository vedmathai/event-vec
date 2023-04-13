class RunConfig():
    def __init__(self):
        self._is_train = False
        self._trainer = None
        self._use_tense = None
        self._llm = None
        self._epochs = None
        self._dataset = None

    def is_train(self):
        return self._is_train
    
    def trainer(self):
        return self._trainer

    def use_tense(self):
        return self._use_tense

    def llm(self):
        return self._llm
    
    def epochs(self):
        return self._epochs
    
    def dataset(self):
        return self._dataset
    
    def set_is_train(self, is_train):
        self._is_train = is_train

    def set_trainer(self, trainer):
        self._trainer = trainer

    def set_use_tense(self, use_tense):
        self._use_tense = use_tense

    def set_llm(self, llm):
        self._llm = llm

    def set_epochs(self, epochs):
        self._epochs = epochs

    def set_dataset(self, dataset):
        self._dataset = dataset

    @staticmethod
    def from_dict(run_config_dict):
        run_config = RunConfig()
        run_config.set_is_train(run_config_dict['is_train'])
        run_config.set_trainer(run_config_dict['trainer'])
        run_config.set_llm(run_config_dict['llm'])
        run_config.set_use_tense(run_config_dict['use_tense'])
        run_config.set_epochs(run_config_dict['epochs'])
        run_config.set_dataset(run_config_dict['dataset'])
        return run_config

    def to_dict(self):
        return {
            'is_train': self.is_train(),
            'trainer': self.trainer(),
            'llm': self.llm(),
            'use_tense': self.use_tense(),
            'epochs': self.epochs(),
            'dataset': self.dataset(),
        }
