
import random
import torch
import json

from eventvec.server.model.event_models.event_model import key_fn


PREPOSITIONS_FILE = 'eventvec/server/data/timebank_prepositions.json'
WORDS_FILE = 'local/data/word2index.txt'
PREP_LIST = ['AFTER', 'BEFORE', 'DURING']


class DataHandler():
    def __init__(self, device):
        self._word2index = {}
        self._index2word = {}
        self._categories = []
        self._device = device

    def load(self):
        self.load_categories()
        self.generate_word2index()

    def load_categories(self):
        with open(PREPOSITIONS_FILE) as f:
            prepositions = json.load(f)
            relationships = set()
            for item in prepositions.values():
                for key in item.keys():
                    relationships.add(key)
            self._categories = sorted(list(relationships))
        self._categories = sorted(PREP_LIST)

    def generate_word2index(self):
        with open(WORDS_FILE) as f:
            for row_i, row in enumerate(f):
                word = row.strip()
                self._word2index[word] = row_i
                self._index2word[row_i] = word
            self._word2index['<UNKNOWN>'] = row_i + 1
            self._index2word[row_i + 1] = '<UNKNOWN>'

    def randomChoice(self, l):
        return l[random.randint(0, len(l) - 1)]

    def randomTrainingPair(self):
        file_name = self.randomChoice(self._file_name2file)
        events = self._file_name2event_set[file_name]
        event_1_idx = random.randint(0, len(events) - 1)
        event_2_idx = event_1_idx + 1
        return (event_1_idx, event_2_idx)

    def categoryTensor(self, category_distribution):
        tensor = torch.zeros(1, len(self._categories), device=self._device)
        for category, category_prob in category_distribution.items():
            li = self._categories.index(category)
            tensor[0][li] = category_prob
        return tensor

    def indexesFromPhrase(self, phrase):
        unknown_index =self._word2index['<UNKNOWN>']
        return [self._word2index.get(word, unknown_index) for word in phrase]

    def tensorFromPhrase(self, phrase):
        indexes = self.indexesFromPhrase(phrase)
        return torch.tensor(indexes, dtype=torch.long, device=self._device).view(-1, 1)

    def inputTensor(self, phrase):
        tensor = self.tensorFromPhrase(phrase)
        return tensor

    def scoreTensor(self, score):
        return torch.tensor([[score]], device=self._device)

    def targetTensor(self, category_distribution):
        category_tensor = self.categoryTensor(category_distribution)
        return category_tensor

    def check_if_empty(self, segment):
        if len(segment) == 0:
            return ['<UNKWOWN>']
        else:
            return segment

    def set_event_input_tensors(self, event):
        verb_segment = [i.orth() for i in sorted(event.verb_nodes(), key=key_fn)]
        verb_segment = self.check_if_empty(verb_segment)
        verb_tensor = self.inputTensor(verb_segment)
        event.set_verb_tensor(verb_tensor)
        object_segment = [i.orth() for i in sorted(event.object_nodes(), key=key_fn)]
        object_segment = self.check_if_empty(object_segment)
        object_tensor = self.inputTensor(object_segment)
        event.set_object_tensor(object_tensor)
        subject_segment = [i.orth() for i in sorted(event.subject_nodes(), key=key_fn)]
        subject_segment = self.check_if_empty(subject_segment)
        subject_tensor = self.inputTensor(subject_segment)
        event.set_subject_tensor(subject_tensor)
        date_segment = [i.orth() for i in sorted(event.date_nodes(), key=key_fn)]
        date_segment = self.check_if_empty(date_segment)
        date_tensor = self.inputTensor(date_segment)
        event.set_date_tensor(date_tensor)


    def n_words(self):
        return len(self._word2index.keys())

    def n_categories(self):
        return len(self._categories)

    def categories(self):
        return self._categories
