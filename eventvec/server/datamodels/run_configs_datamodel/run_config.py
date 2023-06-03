class RunConfig():
    def __init__(self):
        self._id = None
        self._is_train = False
        self._trainer = None
        self._use_tense = None
        self._use_aspect = None
        self._llm = None
        self._epochs = None
        self._dataset = None
        self._use_question_event = None
        self._use_question_event_features = None
        self._use_question_event_aspect = None
        self._use_pos = None
        self._use_question_event_part_of_speech = None
        self._use_negation = None
        self._forward_type = None, 
        self._use_root_verb = None
        self._use_question_classification = None
        self._use_best_of_annotators = None

    def id(self):
        return self._id

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
    
    def use_question_event_features(self):
        return self._use_question_event_features
    
    def use_negation(self):
        return self._use_negation
    
    def use_pos(self):
        return self._use_pos
    
    def use_root_verb(self):
        return self._use_root_verb
    
    def use_question_classification(self):
        return self._use_question_classification
    
    def forward_type(self):
        return self._forward_type
    
    def use_best_of_annotators(self):
        return self._use_best_of_annotators
    
    def set_id(self, id):
        self._id = id

    def set_is_train(self, is_train):
        self._is_train = is_train

    def set_trainer(self, trainer):
        self._trainer = trainer

    def set_use_tense(self, use_tense):
        self._use_tense = use_tense

    def set_use_aspect(self, use_aspect):
        self._use_aspect = use_aspect

    def set_use_pos(self, use_pos):
        self._use_pos = use_pos

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
    
    def set_use_question_event_features(self, use_question_event_features):
        self._use_question_event_features = use_question_event_features

    def set_use_negation(self, use_negation):
        self._use_negation = use_negation

    def set_use_root_verb(self, use_root_verb):
        self._use_root_verb = use_root_verb

    def set_use_question_classification(self, use_question_classification):
        self._use_question_classification = use_question_classification

    def set_forward_type(self, forward_type):
        self._forward_type = forward_type

    def set_use_best_of_annotators(self, use_best_of_annotators):
        self._use_best_of_annotators = use_best_of_annotators

    @staticmethod
    def from_dict(run_config_dict):
        run_config = RunConfig()
        run_config.set_id(run_config_dict['id'])
        run_config.set_is_train(run_config_dict['is_train'])
        run_config.set_trainer(run_config_dict['trainer'])
        run_config.set_llm(run_config_dict['llm'])
        run_config.set_use_tense(run_config_dict['use_tense'])
        run_config.set_use_aspect(run_config_dict['use_aspect'])
        run_config.set_use_pos(run_config_dict['use_pos'])
        run_config.set_use_negation(run_config_dict['use_negation'])
        run_config.set_epochs(run_config_dict['epochs'])
        run_config.set_dataset(run_config_dict['dataset'])
        run_config.set_use_question_event(run_config_dict['use_question_event'])
        run_config.set_use_question_event_features(run_config_dict['use_question_event_features'])
        run_config.set_use_root_verb(run_config_dict['use_root_verb'])
        run_config.set_use_question_classification(run_config_dict['use_question_classification'])
        run_config.set_forward_type(run_config_dict['forward_type'])
        run_config.set_use_best_of_annotators(run_config_dict['use_best_of_annotators'])
        return run_config

    def to_dict(self):
        return {
            'id': self.id(),
            'is_train': self.is_train(),
            'trainer': self.trainer(),
            'llm': self.llm(),
            'use_tense': self.use_tense(),
            'use_aspect': self.use_aspect(),
            'use_pos': self.use_pos(),
            'use_negation': self.use_negation(),
            'epochs': self.epochs(),
            'dataset': self.dataset(),
            'use_question_event': self.use_question_event(),
            'use_question_event_features': self.use_question_event_features(),
            'use_root_verb': self.use_root_verb(),
            'use_question_classification': self.use_question_classification(),
            "forward_type": self.forward_type(),
            "use_best_of_annotators": self.use_best_of_annotators(),
        }
