class RunConfig():
    def __init__(self):
        self._is_train = False
        self._trainer = None
        self._use_tense = None
        self._use_aspect = None
        self._llm = None
        self._epochs = None
        self._dataset = None
        self._use_question_event = None
        self._use_question_event_tense = None
        self._use_question_event_aspect = None
        self._use_part_of_speech = None
        self._use_question_event_part_of_speech = None
        self._use_negation = None
        self._use_root_verb = None
        self._use_question_classification = None

    def is_train(self):
        return self._is_train
    
    def trainer(self):
        return self._trainer

    def use_tense(self):
        return self._use_tense
    
    def use_aspect(self):
        return self._use_aspect

    def llm(self):
        return self._llm
    
    def epochs(self):
        return self._epochs
    
    def dataset(self):
        return self._dataset
    
    def use_question_event(self):
        return self._use_question_event
    
    def use_negation(self):
        return self._use_negation
    
    def use_part_of_speech(self):
        return self._use_part_of_speech
    
    def use_root_verb(self):
        return self._use_root_verb
    
    def use_question_classification(self):
        return self._use_question_classification

    def set_is_train(self, is_train):
        self._is_train = is_train

    def set_trainer(self, trainer):
        self._trainer = trainer

    def set_use_tense(self, use_tense):
        self._use_tense = use_tense

    def set_use_aspect(self, use_aspect):
        self._use_aspect = use_aspect

    def set_use_part_of_speech(self, use_part_of_speech):
        self._use_part_of_speech = use_part_of_speech

    def set_use_negation(self, use_negation):
        self._use_negation = use_negation

    def set_llm(self, llm):
        self._llm = llm

    def set_epochs(self, epochs):
        self._epochs = epochs

    def set_dataset(self, dataset):
        self._dataset = dataset

    def set_use_question_event(self, use_question_event):
        self._use_question_event = use_question_event
    
    def set_use_negation(self, use_negation):
        self._use_negation = use_negation

    def set_use_root_verb(self, use_root_verb):
        self._use_root_verb = use_root_verb

    def set_use_question_classification(self, use_question_classification):
        self._use_question_classification = use_question_classification
    
    @staticmethod
    def from_dict(run_config_dict):
        run_config = RunConfig()
        run_config.set_is_train(run_config_dict['is_train'])
        run_config.set_trainer(run_config_dict['trainer'])
        run_config.set_llm(run_config_dict['llm'])
        run_config.set_use_tense(run_config_dict['use_tense'])
        run_config.set_use_aspect(run_config_dict['use_aspect'])
        run_config.set_use_part_of_speech(run_config_dict['use_part_of_speech'])
        run_config.set_use_negation(run_config_dict['use_negation'])
        run_config.set_epochs(run_config_dict['epochs'])
        run_config.set_dataset(run_config_dict['dataset'])
        run_config.set_use_question_event(run_config_dict['use_question_event'])
        run_config.set_use_root_verb(run_config_dict['use_root_verb'])
        run_config.set_use_question_classification(run_config_dict['use_question_classification'])
        return run_config

    def to_dict(self):
        return {
            'is_train': self.is_train(),
            'trainer': self.trainer(),
            'llm': self.llm(),
            'use_tense': self.use_tense(),
            'use_aspect': self.use_aspect(),
            'use_part_of_speech': self.use_part_of_speech(),
            'use_negation': self.use_negation(),
            'epochs': self.epochs(),
            'dataset': self.dataset(),
            'use_question_event': self.use_question_event(),
            'use_root_verb': self.use_root_verb(),
            'use_question_classification': self.use_question_classification()
        }
