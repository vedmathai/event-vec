
import random
import torch
import json

from eventvec.server.model.event_models.event_model import key_fn


PREPOSITIONS_FILE = 'eventvec/server/data/timebank_prepositions.json'
WORDS_FILE = 'local/data/word2index.txt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DataHandler():
    def __init__(self):
        self._word2index = {}
        self._index2word = {}
        self._categories = []

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

    def categoryTensor(self, category):
        li = self._categories.index(category)
        tensor = torch.zeros(1, len(self._categories), device=device)
        tensor[0][li] = 1
        return tensor

    def indexesFromPhrase(self, phrase):
        unknown_index =self._word2index['<UNKNOWN>']
        return [self._word2index.get(word, unknown_index) for word in phrase]

    def tensorFromPhrase(self, phrase):
        indexes = self.indexesFromPhrase(phrase)
        return torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)

    def inputTensor(self, phrase):
        tensor = self.tensorFromPhrase(phrase)
        return tensor

    def scoreTensor(self, score):
        return torch.tensor([[score]], device=device)

    def targetTensor(self, category, score):
        category_tensor = self.categoryTensor(category)
        score_tensor = self.scoreTensor(score)
        return category_tensor, score_tensor

    def set_event_input_tensors(self, event):
        verb_segment = [i.orth() for i in sorted(event._verb_nodes, key=key_fn)]
        verb_tensor = self.inputTensor(verb_segment)
        event.set_verb_tensor(verb_tensor)
        object_segment = [i.orth() for i in sorted(event._object_nodes, key=key_fn)]
        object_tensor = self.inputTensor(object_segment)
        event.set_object_tensor(object_tensor)
        subject_segment = [i.orth() for i in sorted(event._subject_nodes, key=key_fn)]
        subject_tensor = self.inputTensor(subject_segment)
        event.set_subject_tensor(subject_tensor)

    def n_words(self):
        return len(self._word2index.keys())

    def n_categories(self):
        return len(self._categories)
