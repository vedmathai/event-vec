import random

from eventvec.server.tasks.subordinate.datareader.datareader import SubordinateTemporalDatareader


class SubordinateTemporalDatahandler():
    def __init__(self):
        self._data_readers = {
            'subordinate': SubordinateTemporalDatareader(),
        }

    def load(self, run_config):
        if run_config.dataset() == 'random_split':
            self.load_random_split(run_config)
        elif run_config.dataset() in ['test_past_subordinate', 'test_present_subordinate', 'test_future_subordinate']:
            self.load_by_test_tense_subordinate(run_config)
        elif run_config.dataset() == 'by_sentence':
            self.load_by_sentence(run_config)
        elif run_config.dataset() in ['test_past_matrix', 'test_present_matrix', 'test_future_matrix']:
            self.load_by_test_tense_matrix(run_config)

    def load_random_split(self, run_config):
        data_reader = self._data_readers['subordinate']
        data_said = data_reader.data('temporal_subordinate_said')
        data_stated = data_reader.data('temporal_subordinate_stated')
        data_suggested = data_reader.data('temporal_subordinate_suggested')
        data_insinuated = data_reader.data('temporal_subordinate_insinuated')
        total = len(data_said)
        train_data = []
        test_data = []
        train_choices = random.sample(range(total), int(total*.8))
        test_choices = list(set(range(total)) - set(train_choices))
        for dataset in [data_said, data_stated, data_suggested, data_insinuated]:
            train_data.extend([dataset[i] for i in train_choices])
            test_data.extend([dataset[i] for i in test_choices])
        random.shuffle(train_data)
        random.shuffle(test_data)
        self._train_data = train_data
        self._test_data = test_data

    def load_by_test_tense_subordinate(self, run_config):
        data_reader = self._data_readers['subordinate']
        data_said = data_reader.data('temporal_subordinate_said')
        data_stated = data_reader.data('temporal_subordinate_stated')
        data_suggested = data_reader.data('temporal_subordinate_suggested')
        data_insinuated = data_reader.data('temporal_subordinate_insinuated')
        total = len(data_said)
        train_data = []
        test_data = []
        test_choices = []
        if run_config.dataset() == 'test_past_subordinate':
            tense = 'past'
        elif run_config.dataset() == 'test_present_subordinate':
            tense = 'present'
        elif run_config.dataset() == 'test_future_subordinate':
            tense = 'future'
        for ii, i in enumerate(data_said):
            if i.sub_tense() == tense:
                test_choices += [ii]
        train_choices = list(set(range(total)) - set(test_choices))
        for dataset in [data_said, data_stated, data_suggested, data_insinuated]:
            train_data.extend([dataset[i] for i in train_choices])
            test_data.extend([dataset[i] for i in test_choices])
        random.shuffle(train_data)
        random.shuffle(test_data)
        self._train_data = train_data
        self._test_data = test_data

    def load_by_sentence(self, run_config):
        data_reader = self._data_readers['subordinate']
        data_said = data_reader.data('temporal_subordinate_said')
        data_stated = data_reader.data('temporal_subordinate_stated')
        data_suggested = data_reader.data('temporal_subordinate_suggested')
        data_insinuated = data_reader.data('temporal_subordinate_insinuated')
        train_data = data_stated + data_suggested + data_insinuated
        test_data = data_said
        random.shuffle(train_data)
        random.shuffle(test_data)
        self._train_data = train_data
        self._test_data = test_data

    def load_by_test_tense_matrix(self, run_config):
        data_reader = self._data_readers['subordinate']
        data_said = data_reader.data('temporal_subordinate_said')
        data_stated = data_reader.data('temporal_subordinate_stated')
        data_suggested = data_reader.data('temporal_subordinate_suggested')
        data_insinuated = data_reader.data('temporal_subordinate_insinuated')
        total = len(data_said)
        train_data = []
        test_data = []
        test_choices = []
        if run_config.dataset() == 'test_past_matrix':
            tense = 'Past'
        elif run_config.dataset() == 'test_present_matrix':
            tense = 'Present'
        elif run_config.dataset() == 'test_future_matrix':
            tense = 'Future'
        for ii, i in enumerate(data_said):
            if i.matrix_tense() == tense:
                test_choices += [ii]
        train_choices = list(set(range(total)) - set(test_choices))
        for dataset in [data_said, data_stated, data_suggested, data_insinuated]:
            train_data.extend([dataset[i] for i in train_choices])
            test_data.extend([dataset[i] for i in test_choices])
        random.shuffle(train_data)
        random.shuffle(test_data)
        self._train_data = train_data
        self._test_data = test_data

    def train_data(self):
        return self._train_data

    def test_data(self):
        return self._test_data
